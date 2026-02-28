"""
Get input generator for TritonBench addmm type inputs.
"""

import logging
from typing import Any, Callable

import torch
from tritonbench.operator_loader.aten.input_loader import OperatorInputLoader
from tritonbench.utils.triton_op import PRECISION_DTYPE_MAPPING

logger = logging.getLogger(__name__)


class InputLoader(OperatorInputLoader):
    def __init__(self, tritonbench_op: str, input_config: Any):
        super().__init__(tritonbench_op.name, input_config)
        self.op = tritonbench_op

    def get_input_iter(
        self,
    ) -> Callable:
        shapes = [eval(inp)[1] for inp, _cnt in self.operator_db[self.op_name].items()]
        inputs = []
        for entry in shapes:
            M = int(entry["M"])
            N = int(entry["N"])
            K = int(entry["K"])
            strides = eval(entry["strides"])
            dtype = entry["dtype"]
            if len(strides) != 3:
                logger.warning(
                    "Skipping input with %d strides (expected 3): %s",
                    len(strides),
                    strides,
                )
                continue
            if len(strides[0]) != 2 or len(strides[1]) != 2 or len(strides[2]) != 2:
                logger.warning(
                    "Skipping input with non-2D strides: %s",
                    strides,
                )
                continue
            inputs.append(
                {
                    "shapes": (M, K, N),
                    "dtype": dtype,
                    "strides": strides,
                }
            )

        def _inner():
            requires_grad = self.op.requires_grad
            device = self.op.device
            for obj in inputs:
                shapes = obj["shapes"]
                dtype = PRECISION_DTYPE_MAPPING[obj["dtype"]]
                strides = obj["strides"]
                m, k, n = shapes
                original_m = max(m, strides[1][1])
                original_k = max(k, strides[1][0], strides[2][1])
                original_n = max(n, strides[2][0])
                a = torch.randn((m, n), device=device, dtype=dtype).requires_grad_(
                    requires_grad
                )
                mat1 = torch.randn(
                    (original_m, original_k), device=device, dtype=dtype
                ).requires_grad_(requires_grad)
                mat2 = torch.randn(
                    (original_k, original_n), device=device, dtype=dtype
                ).requires_grad_(requires_grad)
                a = a.as_strided((m, n), strides[0])
                mat1 = mat1.as_strided((m, k), strides[1])
                mat2 = mat2.as_strided((k, n), strides[2])
                if self.op.col_major:
                    mat2 = mat2.T.contiguous().T
                yield a, mat1, mat2

        return _inner
