"""
Add tlx tutorial kernels to tritonbench
"""
import functools
import importlib
import os

from common import setup_tritonbench_cwd

setup_tritonbench_cwd()

from tritonbench.utils.triton_op import register_benchmark
from tritonbench.utils.env_utils import is_blackwell, is_h100
from tritonbench.utils.path_utils import add_path

from typing import Any, Dict, List, Optional, Tuple

META_TRITON_PATH = os.environ.get("TRITONBENCH_TRITON_INSTALL_DIR", None)

assert os.path.exists(META_TRITON_PATH), f"Meta Triton path {META_TRITON_PATH} does not exist."

def load_symbol_from_module(module, symbol):
    with add_path(os.path.join(META_TRITON_PATH, "third_party/tlx/tutorials")):
        module = importlib.import_module(module)
    return getattr(module, symbol)

def add_backend_to_tritonbench(op, backend_name, backend_func, tags, enabled) -> Dict[str, Any]:
    return result_dict

runtime_op_list = [
    ("gemm", "tlx_tutorial_matmul", "blackwell-gemm-ws_test", "matmul"),
    ("gemm", "tlx_tutorial_matmul", "hopper-gemm-ws_test", "matmul"),
    ("blackwell_attentions", "tlx_tutorial_fa_ws_pipelined_persistent", "blackwell-fa-ws-pipelined-persistent_test", "attention"),
    ("flash_attention", "tlx_tutorial_fa_ws_pipelined_pingpong", "hopper-fa-ws-pipelined-pingpong_test", "attention"),
]

def load_tlx_tutorial_backends() -> Dict[str, Any]:
    def _reduce_benchmarks(acc: Dict[str, Any], op_tuple: Tuple[str, str, str, str]) -> Dict[str, Any]:
        op, backend, backend_module, backend_func = op_tuple
        if "blackwell" in backend_module:
            enabled = is_blackwell()
        elif "hopper" in backend_module:
            enabled = is_h100()
        else:
            assert False, f"Unknown backend module {backend_module}"
        # adding the backend at runtime, so if it is not enabled, we don't add it
        if not enabled:
            return acc
        func = load_symbol_from_module(backend_module, backend_func)
        register_benchmark(op, backend, func, tags=["tlx"], enabled=True)
        acc.update({
            op: {
                backend: {
                    "tags": ["tlx"],
                }
            }
        })
        return acc
    runtime_operators = functools.reduce(
        _reduce_benchmarks, runtime_op_list, {}
    )
    return runtime_operators
