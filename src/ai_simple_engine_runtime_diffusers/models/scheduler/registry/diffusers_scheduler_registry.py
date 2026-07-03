from ai_simple_engine.resolver_registry import ResolverRegistry
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class DiffusersSchedulerRegistry(
    ResolverRegistry[
        str,
        type[SchedulerMixin]
    ]
):
    """
    Class that will register the different
    Scheduler classes that we have available
    so we can obtain them using their
    identifier.
    """
    
    def key_for(
        self,
        identifier: str
    ) -> str:
        return identifier