"""
TODO: This is unnecessary, but maybe we want
each ModelExecutor being capable of performing
single operations. Check and remove if not.
"""
from ai_simple_engine_diffusion.types.latents import Latents
from ai_simple_engine.models.executor.abstract import ModelExecutor
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.types.image import Image
from abc import ABC, abstractmethod


class VAEModelExecutorAbstract(
    ModelExecutor,
    ABC
):
    """
    *Abstract class*

    To transform images into latents and
    latents into images.
    """

    @abstractmethod
    async def encode(
        self,
        model: LoadedModel,
        image: Image
    ) -> Latents:
        """
        Encode an image into latent space.
        """
        ...

    @abstractmethod
    async def decode(
        self,
        model: LoadedModel,
        latents: Latents
    ) -> Image:
        """
        Decode latent representations into an image.
        """
        ...