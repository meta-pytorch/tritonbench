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
            bias_shape = eval(entry["bias"]) if "bias" in entry else None
            if len(strides) not in (2, 3):
                logger.warning(
                    "Skipping input with %d strides (expected 2 or 3): %s",
                    len(strides),
                    strides,
                )
                continue
            matrix_strides = strides[-2:]
            if any(len(matrix_stride) != 2 for matrix_stride in matrix_strides):
                logger.warning(
                    "Skipping input with non-2D strides: %s",
                    strides,
                )
                continue
            inputs.append(
                {
                    "shapes": (M, K, N),
                    "dtype": dtype,
                    "strides": matrix_strides,
                    "bias_shape": bias_shape,
                    "bias_strides": strides[0] if len(strides) == 3 else None,
                }
            )

        def _inner():
            requires_grad = self.op.requires_grad
            device = self.op.device
            for obj in inputs:
                shapes = obj["shapes"]
                dtype = PRECISION_DTYPE_MAPPING[obj["dtype"]]
                strides = obj["strides"]
                bias_shape = obj["bias_shape"]
                bias_strides = obj["bias_strides"]
                m, k, n = shapes
                original_m = max(m, strides[0][1])
                original_k = max(k, strides[0][0], strides[1][1])
                original_n = max(n, strides[1][0])
                if bias_shape is not None:
                    a = torch.randn(
                        bias_shape, device=device, dtype=dtype
                    ).requires_grad_(requires_grad)
                else:
                    original_a_rows = max(m, bias_strides[1])
                    original_a_cols = max(n, bias_strides[0])
                    a = torch.randn(
                        (original_a_rows, original_a_cols),
                        device=device,
                        dtype=dtype,
                    ).requires_grad_(requires_grad)
                    a = a.as_strided((m, n), bias_strides)
                mat1 = torch.randn(
                    (original_m, original_k), device=device, dtype=dtype
                ).requires_grad_(requires_grad)
                mat2 = torch.randn(
                    (original_k, original_n), device=device, dtype=dtype
                ).requires_grad_(requires_grad)
                mat1 = mat1.as_strided((m, k), strides[0])
                mat2 = mat2.as_strided((k, n), strides[1])
                if self.op.col_major:
                    mat2 = mat2.T.contiguous().T
                yield a, mat1, mat2

        return _inner
