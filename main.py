import copy
import functools
import logging
import os
import random
import typing

import click
import diffusers
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
)
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import make_image_grid
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import get_token
import torchvision.transforms.functional as TF
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

import wandb
from utils.webdataset import Text2ImageDataset

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


def preprocess(image):
    image = TF.resize(image, 512)

    # get crop coordinates and crop image
    c_top, c_left, _, _ = transforms.RandomCrop.get_params(
        image, output_size=(512, 512)
    )
    image = TF.crop(image, c_top, c_left, 512, 512)
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.5], [0.5])

    return image


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

    return pixel_values, prompts


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
@click.option("--dataset-name", type=str)
@click.option("--webdataset-url", type=str)
@click.option(
    "--webdataset-epoch-samples",
    type=int,
    default=1_000_000,
    help="Maximum number of samples per epoch. Useful when your data is large and you want to randomly select a subset each epoch.",
)
@click.option("--batch-size", type=int, default=4)
@click.option("--num-epochs", type=int, default=10)
@click.option("--learning-rate", type=float, default=1e-6)
@click.option("--image-column", type=str, default="image")
@click.option("--text-column", type=str, default="text")
@click.option(
    "--instance-prompt", type=str, help="Single prompt to use for all examples"
)
def main(
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    dataset_name: typing.Optional[str],
    webdataset_url: typing.Optional[str],
    webdataset_epoch_samples: int,
    image_column: str,
    text_column: str,
    instance_prompt: typing.Optional[str],
):
    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="vae"
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="scheduler",
        **{
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "num_train_timesteps": 100,
            "shift": 3.0,
            "use_dynamic_shifting": True,
        },
    )

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder_2 = T5EncoderModel.from_pretrained("google-t5/t5-base")
    tokenizer_2 = T5TokenizerFast.from_pretrained("google-t5/t5-base")

    transformer = FluxTransformer2DModel(
        **{
            "attention_head_dim": 128,
            "guidance_embeds": True,
            "in_channels": 64,
            "joint_attention_dim": 768,
            "num_attention_heads": 4,
            "num_layers": 19,
            "num_single_layers": 38,
            "patch_size": 1,
            "pooled_projection_dim": 768,
        }
    )

    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2 = text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    if dataset_name is not None:
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

    if webdataset_url is not None:
        train_dataset = Text2ImageDataset(
            train_shards_path_or_url=webdataset_url,
            num_train_examples=webdataset_epoch_samples,
            per_gpu_batch_size=batch_size,
            global_batch_size=batch_size * accelerator.num_processes,
            num_workers=1,
            resolution=512,
            shuffle_buffer_size=10000,
            pin_memory=True,
            persistent_workers=True,
        )
        train_dataloader = train_dataset.train_dataloader

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
        eps=1e-08,
    )

    total_steps = (len(train_dataloader) * num_epochs) * accelerator.num_processes
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=total_steps * 0.05,
        num_training_steps=total_steps,
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

    for _ in range(num_epochs):
        for _, (images, prompts) in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                model_input_original = vae.encode(
                    images.to(device=accelerator.device, dtype=vae.dtype)
                ).latent_dist.sample()

                model_input = (
                    model_input_original - vae.config.shift_factor
                ) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

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
                    weighting_scheme=None,
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

                guidance = torch.tensor(
                    [random.uniform(1.0, 20.0)], device=accelerator.device
                )
                guidance = guidance.expand(model_input.shape[0])

                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    [text_encoder, text_encoder_2],
                    [tokenizer, tokenizer_2],
                    prompts,
                    256,
                )

                text_ids = text_ids.to(device=accelerator.device)

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
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
                    weighting_scheme=None, sigmas=sigmas
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

                    if global_step % 100 == 0:
                        autocast_ctx = torch.autocast(accelerator.device.type)

                        with autocast_ctx:
                            with torch.no_grad():
                                original_noised_decoded = vae.decode(
                                    noisy_model_input, return_dict=False
                                )[0]
                                original_decoded = vae.decode(
                                    model_input_original, return_dict=False
                                )[0]
                                predicted_decoded = vae.decode(
                                    (noisy_model_input - model_pred).detach(),
                                    return_dict=False,
                                )[0]

                            all = torch.cat(
                                [
                                    original_noised_decoded[:4, :, :, :],
                                    original_decoded[:4, :, :, :],
                                    predicted_decoded[:4, :, :, :],
                                ],
                                0,
                            )

                            images = image_processor.postprocess(
                                all.cpu(), output_type="pil"
                            )

                            image_grid = make_image_grid(images, rows=3, cols=4)

                            for tracker in accelerator.trackers:
                                phase_name = "test"
                                tracker.log({phase_name: [wandb.Image(image_grid)]})

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
