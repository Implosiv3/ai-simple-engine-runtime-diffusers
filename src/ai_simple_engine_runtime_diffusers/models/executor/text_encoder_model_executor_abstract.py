"""
TODO: This is unnecessary, but maybe we want
each ModelExecutor being capable of performing
single operations. Check and remove if not.
"""
from ai_simple_engine.models.executor.abstract import ModelExecutor
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine_diffusion.types.data_type import Embeddings
from abc import ABC, abstractmethod


# This will be to perform 'EncodePrompt()'
class TextEncoderModelExecutorAbstract(
    ModelExecutor,
    ABC
):
    """
    *Abstract class*

    To convert text into embeddings
    """

    @abstractmethod
    async def encode(
        self,
        model: LoadedModel,
        prompt: str
    ) -> Embeddings:
        """
        Encode the given prompt into embeddings.
        """
        ...