from ai_simple_engine.device.base import Device

import torch


def get_torch_dtype_for(
    device: Device
):
    """
    Get the `torch` `dtype` for the `device`
    provided.
    """
    if device.type == 'cuda':
        return torch.float16

    if device.type == 'mps':
        return torch.float16

    return torch.float32