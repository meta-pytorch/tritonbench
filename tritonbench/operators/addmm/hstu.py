import importlib
from typing import Tuple

import torch
import triton
from tritonbench.utils.path_utils import add_path, get_hstu_path

with add_path(str(get_hstu_path())):
    from generative_recommenders.ops.triton.triton_addmm import _AddMmFunction


@torch.fx.wrap
def triton_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    return _AddMmFunction.apply(mat1, mat2, input)
