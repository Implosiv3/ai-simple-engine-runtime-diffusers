from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL
from dataclasses import dataclass


@dataclass(frozen = True)
class DiffusersLatentDiffusionModel:
    """
    Runtime representation of a Latent Diffusion
    model loaded through Diffusers.
    """

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    scheduler_config: dict

    