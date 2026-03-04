"""
Get input generator for TritonBench gemm type inputs.
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
            if len(strides) != 2:
                logger.warning(
                    "Skipping input with %d strides (expected 2): %s",
                    len(strides),
                    strides,
                )
                continue
            if len(strides[0]) != 2 or len(strides[1]) != 2:
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
                # The shape might from a tensor view, which is different from the original shape
                # Try to infer the original shape from both shape and strides
                actual_m = max(m, strides[0][1])
                actual_k = max(k, strides[0][0], strides[1][1])
                actual_n = max(n, strides[1][0])
                # Skip shapes whose backing storage exceeds GPU memory
                elem_size = torch.tensor([], dtype=dtype).element_size()
                total_bytes = (actual_m * actual_k + actual_k * actual_n) * elem_size
                free_mem, _ = torch.cuda.mem_get_info(device)
                if total_bytes > free_mem:
                    print(
                        f"Skipping shape (M={m}, K={k}, N={n}) with strides {strides}: requires {total_bytes / 2**30:.1f} GiB but GPU only has {free_mem / 2**30:.1f} GiB free"
                    )
                    continue
                a = self.op._scaled_randn(
                    (actual_m, actual_k), scale=k, device=device, dtype=dtype
                ).requires_grad_(requires_grad)
                w = self.op._scaled_randn(
                    (actual_k, actual_n), scale=k, device=device, dtype=dtype
                ).requires_grad_(requires_grad)
                a = a.as_strided(size=[m, k], stride=strides[0]).requires_grad_(
                    requires_grad
                )
                w = w.as_strided(size=[k, n], stride=strides[1]).requires_grad_(
                    requires_grad
                )
                yield a, w, None

        return _inner
