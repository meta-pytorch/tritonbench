import torch

# PowerManagerTask requires pynvml which is only available on NVIDIA GPUs.
# torch.version.hip is set on ROCm/AMD builds where pynvml is not available.
if torch.cuda.is_available() and not torch.version.hip:
    from .power_manager import PowerManagerTask
else:
    PowerManagerTask = None
