from ai_simple_engine_runtime_diffusers.models.loader.latent_diffusion_model_loader import LatentDiffusionModelLoader
from ai_simple_engine_runtime_diffusers.models.scheduler.utils import SCHEDULERS
from ai_simple_engine_runtime_diffusers.models.scheduler.registry.diffusers_scheduler_registry import DiffusersSchedulerRegistry
from ai_simple_engine.engine_builder import EngineBuilder
from ai_simple_engine.plugins.plugin import Plugin


class DiffusersRuntimeMusicgenPlugin(
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
            builder.add_model_loader(LatentDiffusionModelLoader)
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
        scheduler_registry = builder.get_or_add_service(DiffusersSchedulerRegistry)

        for identifier, cls in SCHEDULERS.items():
            scheduler_registry.register(
                identifier,
                cls
            )
        
        # scheduler_registry.register(
        #     'euler',
        #     EulerDiscreteScheduler
        # )

        # TODO: This below was the 'musicgen'
        # (
        #     builder
        #     .add_model_loader(MusicgenModelLoader())
        # )

        # """
        # We obtain the registry that handles the
        # model executors by the model's family,
        # and we register our specific model
        # executor that uses transformers.
        # """
        # registry = builder.get_or_add_service(FamilyModelExecutorRegistry)

        # registry.register(
        #     MUSICGEN_MODEL_FAMILY,
        #     TransformersMusicgenModelExecutor()
        # )
