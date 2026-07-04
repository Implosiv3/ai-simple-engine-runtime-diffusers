from ai_simple_engine_diffusion.noise.noise_generator_abstract import NoiseGenerator

import torch


class TorchNoiseGenerator(
    NoiseGenerator
):
    """
    Class to generate random noise using `torch`.
    """

    async def generate(
        self,
        *,
        shape,
        device,
        dtype,
        seed: int
    ):
        generator = torch.Generator(device = device).manual_seed(seed)

        return torch.randn(
            shape,
            generator = generator,
            device = device,
            dtype = dtype,
        )