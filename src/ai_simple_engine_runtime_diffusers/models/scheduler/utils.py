"""
TODO: Maybe rename and move this module.
"""
from diffusers.schedulers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler
)

SCHEDULERS = {
    'ddim': DDIMScheduler,
    'euler': EulerDiscreteScheduler,
    'euler_a': EulerAncestralDiscreteScheduler,
    'dpmpp_2m': DPMSolverMultistepScheduler,
    'flow_match_euler': FlowMatchEulerDiscreteScheduler
}
"""
The list including all the schedulers we are
accepting.
"""


def get_scheduler_class(
    scheduler: str,
    model: LoadedModel
) -> Union[SchedulerMixin, None]:
    scheduler_class = SCHEDULERS.get(scheduler, None)

    if scheduler_class is None:
        raise Exception(f'The scheduler "{scheduler}" is not accepted by our system.')
    
    return scheduler_class

    scheduler_cls = registry.resolve(model.info.scheduler.identifier)

    scheduler = scheduler_cls.from_config(...)

    return scheduler

    return scheduler_cls.from_config(
        model.model.scheduler.config
    )
