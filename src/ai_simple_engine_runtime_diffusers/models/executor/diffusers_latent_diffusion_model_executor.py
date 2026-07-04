from ai_simple_engine_runtime_diffusers.models.scheduler.diffusers_scheduler import DiffusersScheduler
from ai_simple_engine_diffusion.model.executor.abstract.latent_diffusion_model_executor_abstract import LatentDiffusionModelExecutorAbstract
from ai_simple_engine.execution.execution_context import ExecutionContext
from ai_simple_engine_runtime_diffusers.models.executor.prompt_embeddings.prompt_embeddings import PromptEmbeddings
from ai_simple_engine_runtime_diffusers.models.scheduler.registry.diffusers_scheduler_registry import DiffusersSchedulerRegistry
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.types.image import Image

import torch


class DiffusersLatentDiffusionModelExecutor (
    LatentDiffusionModelExecutorAbstract
):
    """
    Latent diffusion executor based on the Hugging Face
    Diffusers library.

    Implements the model-specific operations required by
    `LatentDiffusionModelExecutor` using Diffusers
    components such as the tokenizer, text encoder,
    UNet/DiT, VAE and schedulers.
    """
    
    def create_scheduler(
        self,
        context: ExecutionContext,
        model: LoadedModel,
    ):
        scheduler_registry = context.services.get(DiffusersSchedulerRegistry)

        scheduler_class = scheduler_registry.resolve(model.info.scheduler.identifier)

        scheduler = scheduler_class.from_config(
            model.instance.scheduler_config,
            **model.info.scheduler.kwargs
        )

        return DiffusersScheduler(scheduler)
        

    async def predict_noise(
        self,
        *,
        model: LoadedModel,
        latents,
        embeddings: PromptEmbeddings,
        timestep,
        guidance_scale
    ):
        runtime_model = model.instance
        unet = runtime_model.unet
        latent_input = torch.cat([latents, latents])

        encoder_hidden_states = torch.cat([
            embeddings.negative,
            embeddings.positive
        ])

        noise = unet(
            latent_input,
            timestep,
            encoder_hidden_states = encoder_hidden_states
        ).sample

        noise_uncond, noise_text = noise.chunk(2)

        return noise_uncond + guidance_scale * (
            noise_text - noise_uncond
        )
    
    # Specific methods below
    async def encode_prompt(
        self,
        *,
        model: LoadedModel,
        prompt: str,
        negative_prompt: str
    ) -> PromptEmbeddings:
        runtime_model = model.instance

        tokenizer = runtime_model.tokenizer
        text_encoder = runtime_model.text_encoder

        positive = tokenizer(
            prompt,
            padding = 'max_length',
            truncation = True,
            max_length = tokenizer.model_max_length,
            return_tensors = 'pt'
        )

        negative = tokenizer(
            negative_prompt,
            padding = 'max_length',
            truncation = True,
            max_length = tokenizer.model_max_length,
            return_tensors = 'pt'
        )

        device = next(text_encoder.parameters()).device
        positive = text_encoder(positive.input_ids.to(device))[0]
        negative = text_encoder(negative.input_ids.to(device))[0]

        return PromptEmbeddings(
            positive = positive,
            negative = negative
        )
    
    async def decode_latents(
        self,
        *,
        model: LoadedModel,
        latents
    ) -> Image:
        runtime_model = model.instance

        vae = runtime_model.vae

        latents = latents / vae.config.scaling_factor

        image = vae.decode(latents).sample
        """
        TODO: I don't know if this must be done always
        or depending or what, but I should pay attention
        and refactor it.
        """
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
        image = Image(image)

        return image