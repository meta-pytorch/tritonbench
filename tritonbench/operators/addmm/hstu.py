import importlib
from typing import Tuple

import torch
import triton
from tritonbench.utils.path_utils import add_path, SUBMODULE_PATH

with add_path(str(SUBMODULE_PATH.joinpath("generative-recommenders"))):
    from generative_recommenders.ops.triton.triton_addmm import (
        _AddMmFunction,
        triton_addmm_fwd_tma_persistent,
    )


@torch.fx.wrap
def triton_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    return _AddMmFunction.apply(mat1, mat2, input)


@torch.fx.wrap
def triton_addmm_fwd_b200_direct(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    add_in_bias_dtype: bool = False,
) -> torch.Tensor:
    return triton_addmm_fwd_tma_persistent(
        mat1,
        mat2,
        input,
        add_in_bias_dtype=add_in_bias_dtype,
    )
