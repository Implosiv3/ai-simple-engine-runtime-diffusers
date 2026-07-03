"""
TODO: This is unnecessary, but maybe we want
each ModelExecutor being capable of performing
single operations. Check and remove if not.
"""
from ai_simple_engine_diffusion.types.latents import Latents
from ai_simple_engine_diffusion.types.noise_prediction import NoisePrediction
from ai_simple_engine_diffusion.types.embeddings import Embeddings
from ai_simple_engine.models.executor.abstract import ModelExecutor
from ai_simple_engine.models.loaded_model import LoadedModel
from abc import ABC, abstractmethod


class UNetModelExecutorAbstract(
    ModelExecutor,
    ABC
):
    """
    *Abstract class*

    To predict the noise.
    """

    @abstractmethod
    async def predict_noise(
        self,
        model: LoadedModel,
        latents: Latents,
        embeddings: Embeddings,
        timestep: int
    ) -> NoisePrediction:
        """
        Predict the noise residual for the given
        latents at the specified timestep.
        """
        ...