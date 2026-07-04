class DiffusersScheduler:

    def __init__(
        self,
        scheduler
    ):
        self._scheduler = scheduler

    @property
    def init_noise_sigma(
        self
    ):
        """
        The `gamma` that must be used when starting
        to create the noise.
        """
        return self._scheduler.init_noise_sigma

    @property
    def timesteps(
        self
    ):
        return self._scheduler.timesteps

    def set_timesteps(
        self,
        steps: int
    ):
        self._scheduler.set_timesteps(steps)

    def scale_model_input(
        self,
        latents,
        timestep
    ):
        return self._scheduler.scale_model_input(
            latents,
            timestep
        )
    
    def step(
        self,
        *,
        model_output,
        timestep,
        sample
    ):
        return self._scheduler.step(
            model_output = model_output,
            timestep = timestep,
            sample = sample
        )