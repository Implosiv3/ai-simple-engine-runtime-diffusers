from ai_simple_engine_runtime_diffusers.models.loader.latent_diffusion_model_loader import LatentDiffusionModelLoader
from ai_simple_engine_runtime_diffusers.models.executor.noise.pytorch_noise_generator import TorchNoiseGenerator
from ai_simple_engine_runtime_diffusers.models.scheduler.registry.diffusers_scheduler_registry import DiffusersSchedulerRegistry
from ai_simple_engine.models.executor.registry.family_model_executor_registry import FamilyModelExecutorRegistry
from ai_simple_engine_runtime_diffusers.models.executor.diffusers_latent_diffusion_model_executor import DiffusersLatentDiffusionModelExecutor
from ai_simple_engine_runtime_diffusers.models.scheduler.utils import SCHEDULERS
from ai_simple_engine_diffusion.noise.noise_generator_abstract import NoiseGenerator
from ai_simple_engine.engine_builder import EngineBuilder
from ai_simple_engine.plugins.plugin import Plugin


class DiffusersRuntimePlugin(
    Plugin
):
    """
    The plugin to add the diffusers' models
    functionality.

    This plugin includes:
    """

    def register(
        self,
        builder: EngineBuilder
    ):
        (
            builder.add_model_loader(LatentDiffusionModelLoader())
        )

        (
            builder.add_service(
                NoiseGenerator,
                TorchNoiseGenerator()
            )
        )

        # TODO: Register the SchedulerRegistry
        """
        Dinamycally register all the scheduler classes
        incldued in the 'SCHEDULERS' dict. The executor
        will resolve the instance when trying to
        execute:

        scheduler_cls = registry.resolve(
            scheduler_spec.identifier
        )
        """
        model_executor_registry = builder.get_or_add_service(FamilyModelExecutorRegistry)

        model_executor_registry.register(
            'latent_diffusion',
            DiffusersLatentDiffusionModelExecutor()
        )

        scheduler_registry = builder.get_or_add_service(DiffusersSchedulerRegistry)

        for identifier, cls in SCHEDULERS.items():
            scheduler_registry.register(
                identifier,
                cls
            )