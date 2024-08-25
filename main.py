import copy
import functools
import logging
import os
import random
import typing

import click
import diffusers
import prodigyopt
import torch
import torch.utils.checkpoint
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

logger = get_logger(__name__)


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(
        device=device, dtype=dtype
    )
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def collate_fn(
    examples, image_column: str, text_column: str, instance_prompt: typing.Optional[str]
):
    pixel_values = [preprocess(example[image_column]) for example in examples]

    if instance_prompt is None:
        prompts = [
            (
                random.choice(example[text_column])
                if isinstance(example[text_column], list)
                else example[text_column]
            )
            for example in examples
        ]
    else:
        prompts = [instance_prompt] * len(pixel_values)

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}

    return batch


def log_validation(
    pipeline,
    accelerator,
    pipeline_args,
    validation_prompt,
):
    logger.info(
        f"Running validation... \n Generating 4 images with prompt:"
        f" {validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        images = [pipeline(**pipeline_args).images[0] for _ in range(4)]

    for tracker in accelerator.trackers:
        phase_name = "test"
        tracker.log(
            {
                phase_name: [
                    wandb.Image(image, caption=f"{i}: {validation_prompt}")
                    for i, image in enumerate(images)
                ]
            }
        )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


@click.command()
@click.option("--output-dir", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
@click.option("--batch-size", type=int, default=4)
@click.option("--num-epochs", type=int, default=10)
@click.option("--learning-rate", type=float, default=1e-6)
@click.option("--image-column", type=str, default="image")
@click.option("--text-column", type=str, default="text")
@click.option("--validation-prompt", type=str, default="an apple")
@click.option(
    "--instance-prompt", type=str, help="Single prompt to use for all examples"
)
def main(
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    dataset_name: str,
    image_column: str,
    text_column: str,
    validation_prompt: str,
    instance_prompt: typing.Optional[str],
):
    train_dataset = load_dataset(dataset_name)["train"]
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            collate_fn,
            image_column=image_column,
            text_column=text_column,
            instance_prompt=instance_prompt,
        ),
    )

    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="vae"
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="scheduler",
        num_train_timesteps=100
    )

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder_2 = T5EncoderModel.from_pretrained("google-t5/t5-small")
    tokenizer_2 = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir="logs"
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with="wandb",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    transformer = FluxTransformer2DModel(
        **{
            "attention_head_dim": 128,
            "guidance_embeds": True,
            "in_channels": 64,
            "joint_attention_dim": 512,
            "num_attention_heads": 6,
            "num_layers": 38,
            "num_single_layers": 38,
            "patch_size": 1,
            "pooled_projection_dim": 512,
        }
    )

    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2 = text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), FluxTransformer2DModel):
                    unwrap_model(model).save_pretrained(
                        os.path.join(output_dir, "transformer")
                    )
                elif isinstance(
                    unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)
                ):
                    if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                        unwrap_model(model).save_pretrained(
                            os.path.join(output_dir, "text_encoder")
                        )
                    else:
                        unwrap_model(model).save_pretrained(
                            os.path.join(output_dir, "text_encoder_2")
                        )
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), FluxTransformer2DModel):
                load_model = FluxTransformer2DModel.from_pretrained(
                    input_dir, subfolder="transformer"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
            elif isinstance(
                unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)
            ):
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(
                        input_dir, subfolder="text_encoder"
                    )
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = T5EncoderModel.from_pretrained(
                            input_dir, subfolder="text_encoder_2"
                        )
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        raise ValueError(
                            f"Couldn't load the model of type: ({type(model)})."
                        )
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer.parameters(),
        "lr": learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]

    optimizer_class = torch.optim.AdamW
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-04,
        eps=1e-08
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=(len(train_dataloader) * num_epochs) * 0.05,
        num_training_steps=(len(train_dataloader) * num_epochs)
        * accelerator.num_processes,
        num_cycles=1,
        power=1.0,
    )


    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        tracker_name = "tiny-flux"
        accelerator.init_trackers(tracker_name)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    progress_bar = tqdm(
        range(0, len(train_dataloader) * num_epochs),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0

    for epoch in range(num_epochs):
        for step, sample in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                model_input = vae.encode(
                    sample["pixel_values"].to(dtype=vae.dtype)
                ).latent_dist.sample()

                model_input = (
                    model_input - vae.config.shift_factor
                ) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2],
                    model_input.shape[3],
                    accelerator.device,
                    weight_dtype,
                )
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme="mode",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=model_input.device
                )

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(
                    timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                )
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                if transformer.config.guidance_embeds:
                    guidance = torch.tensor(
                        [random.uniform(1.0, 10.0)], device=accelerator.device
                    )
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    [text_encoder, text_encoder_2],
                    [tokenizer, tokenizer_2],
                    sample["prompts"],
                    77,
                )

                text_ids = text_ids.to(device=accelerator.device)

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[2] * vae_scale_factor / 2),
                    width=int(model_input.shape[3] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="mode", sigmas=sigmas
                )

                # flow matching loss
                target = noise - model_input

                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % 100 == 0 and global_step > 0:
                        with torch.no_grad():
                            pipeline = FluxPipeline.from_pretrained(
                                "black-forest-labs/FLUX.1-schnell",
                                text_encoder=accelerator.unwrap_model(text_encoder),
                                text_encoder_2=accelerator.unwrap_model(text_encoder_2),
                                transformer=accelerator.unwrap_model(transformer),
                                scheduler=noise_scheduler,
                                tokenizer=tokenizer,
                                tokenizer_2=tokenizer_2,
                                torch_dtype=weight_dtype,
                            )

                            pipeline_args = {
                                "prompt": validation_prompt,
                                "width": 256,
                                "height": 256,
                            }
                            log_validation(
                                pipeline=pipeline,
                                accelerator=accelerator,
                                pipeline_args=pipeline_args,
                                validation_prompt=(
                                    validation_prompt
                                    if instance_prompt is None
                                    else instance_prompt
                                ),
                            )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)

        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
        )

        pipeline.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
