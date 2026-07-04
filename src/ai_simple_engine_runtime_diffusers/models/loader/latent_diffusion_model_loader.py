from ai_simple_engine_runtime_diffusers.models.runtime.diffusers_latent_diffusion_model import DiffusersLatentDiffusionModel
from ai_simple_engine_runtime_diffusers.models.runtime.utils import get_torch_dtype_for
from ai_simple_engine_diffusion.model.info.diffusion_model_info import DiffusionModelInfo
from ai_simple_engine_diffusion.model.info.tensor_output_spec import TensorOutputSpec
from ai_simple_engine_diffusion.scheduler.spec.base import SchedulerSpec
from ai_simple_engine.device.base import Device
from ai_simple_engine.models.loaders.abstract import ModelLoader
from ai_simple_engine.models.installed_model import InstalledModel
from ai_simple_engine.models.loaded_model import LoadedModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL


class LatentDiffusionModelLoader(
    ModelLoader
):
    
    @property
    def family(
        self
    ) -> str:
        # TODO: Transform into const
        return 'latent_diffusion'

    async def load(
        self,
        installed_model: InstalledModel,
        device: Device
    ) -> LoadedModel:
        path = str(installed_model.path)

        tokenizer = CLIPTokenizer.from_pretrained(
            path,
            subfolder = 'tokenizer',
            torch_dtype = get_torch_dtype_for(device)
        )#.to(str(device))

        text_encoder = CLIPTextModel.from_pretrained(
            path,
            subfolder = 'text_encoder',
            torch_dtype = get_torch_dtype_for(device)
        ).to(str(device))

        unet = UNet2DConditionModel.from_pretrained(
            path,
            subfolder = 'unet',
            torch_dtype = get_torch_dtype_for(device)
        ).to(str(device))

        vae = AutoencoderKL.from_pretrained(
            path,
            subfolder = 'vae',
            torch_dtype = get_torch_dtype_for(device)
        ).to(str(device))

        # TODO: This is if I want its own Scheduler
        # DDIMScheduler.from_pretrained(
        #     path,
        #     subfolder = 'scheduler',
        #     torch_dtype = get_torch_dtype_for(device)
        # )#.to(str(device))

        # TODO: We had this before but was failing
        scheduler_config = AutoencoderKL.load_config(
            path + '/scheduler/scheduler_config.json',
            subfolder = 'scheduler',
            torch_dtype = get_torch_dtype_for(device)
        )#.to(str(device))

        # Free memory
        # TODO: Explain why (?)
        unet.eval()
        text_encoder.eval()
        vae.eval()

        runtime_model = DiffusersLatentDiffusionModel(
            tokenizer = tokenizer,
            text_encoder = text_encoder,
            unet = unet,
            vae = vae,
            scheduler_config = scheduler_config
        )

        info = DiffusionModelInfo(
            latent_channels = unet.config.in_channels,
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1),
            # TODO: Make dynamic depending on the 'installed_model'
            scheduler = self._default_scheduler(installed_model),
            # TODO: Is this ok (?)
            tensor_output_spec = TensorOutputSpec(
                value_range = (-1.0, 1.0)
            )
        )

        return LoadedModel(
            installed_model = installed_model,
            instance = runtime_model,
            info = info
        )
    
    def _default_scheduler(
        self,
        installed_model: InstalledModel
    ) -> SchedulerSpec:
        """
        *For internal use only*

        Get the scheduler by default for the
        `installed_model` provided.
        """
        # TODO: Make dynamic based on the 'installed_model'
        return SchedulerSpec(
            identifier = 'euler'
        )