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