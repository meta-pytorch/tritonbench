import logging
import os
import subprocess
import sys
from pathlib import Path

from tools.python_utils import get_pip_cmd


logger = logging.getLogger(__name__)

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent


def install_int21():
    """Build the bundled INT21 RMSNorm CUDA (sm_100a) kernel and export it as a
    PyTorch custom operator.

    The kernel lives at ``tritonbench/kernels/int21/rmsnorm_b200.cu`` and is
    JIT-compiled via ``torch.utils.cpp_extension.load``. Importing the module
    registers the ``int21::rmsnorm_fwd_out`` / ``int21::rmsnorm_bwd_out`` custom
    operators, and ``load_ext()`` forces the compilation up front so the cached
    artifact is ready before benchmarking.
    """
    # ``pybind11`` and ``ninja`` are required by the JIT extension build.
    cmd = get_pip_cmd() + ["install", "pybind11", "ninja"]
    subprocess.check_call(cmd)

    sys.path.insert(0, str(REPO_PATH))
    from tritonbench.kernels.int21.rmsnorm_b200 import load_ext

    logger.info("[tritonbench] building int21 rmsnorm_b200.cu (sm_100a)...")
    load_ext()
    logger.info("[tritonbench] int21 RMSNorm custom operator is ready.")
