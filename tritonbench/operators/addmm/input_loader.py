"""
Get input generator for TritonBench addmm type inputs.
"""

from typing import Any, Callable

import torch
from tritonbench.operator_loader.aten.input_loader import OperatorInputLoader
from tritonbench.utils.triton_op import PRECISION_DTYPE_MAPPING


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
            assert len(strides) == 3, (
                f"Can only have 3 strides from input, get: {strides}"
            )
            assert (
                len(strides[0]) == 2 and len(strides[1]) == 2 and len(strides[2]) == 2
            ), f"Can only deal with 2D strides, get: {strides}"
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
