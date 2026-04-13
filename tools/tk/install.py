import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
TK_PATH = REPO_PATH.joinpath("submodules", "ThunderKittens")
TK_TOOLS_PATH = REPO_PATH.joinpath("tools", "tk")
TK_BUILD_PATH = REPO_PATH.joinpath("build")
TK_PACKAGE_PATH = TK_BUILD_PATH.joinpath("thunderkittens")

PACKAGE_SOURCES = {
    "__init__.py": """from .bf16_b200 import bf16_b200_gemm
from .fp8_h100 import fp8_gemm
from .mha_h100 import mha_backward, mha_forward

__all__ = [
    "bf16_b200_gemm",
    "fp8_gemm",
    "mha_backward",
    "mha_forward",
]
""",
    "_runtime.py": """import ctypes
from pathlib import Path

import torch

_PRELOADED = False


def preload_torch_deps():
    global _PRELOADED
    if _PRELOADED:
        return

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    for lib_name in (
        "libc10.so",
        "libc10_cuda.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libtorch_python.so",
    ):
        lib_path = torch_lib_dir / lib_name
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    _PRELOADED = True
""",
    "bf16_b200/__init__.py": """from .._runtime import preload_torch_deps

preload_torch_deps()

from ._C import bf16_b200_gemm as _bf16_b200_gemm_impl


def bf16_b200_gemm(a, b):
    return _bf16_b200_gemm_impl(a, b.transpose(0, 1).contiguous())


__all__ = ["bf16_b200_gemm"]
""",
    "fp8_h100/__init__.py": """from .._runtime import preload_torch_deps

preload_torch_deps()

from ._C import fp8_gemm as _fp8_gemm_impl


def fp8_gemm(a, b):
    return _fp8_gemm_impl(a, b.transpose(0, 1).contiguous())


__all__ = ["fp8_gemm"]
""",
    "mha_h100/__init__.py": """from .._runtime import preload_torch_deps

preload_torch_deps()

from ._C import mha_backward, mha_forward

__all__ = ["mha_backward", "mha_forward"]
""",
}


def _get_env():
    environ = os.environ.copy()
    build_path = str(TK_BUILD_PATH)
    if environ.get("PYTHONPATH"):
        environ["PYTHONPATH"] = f"{build_path}:{environ['PYTHONPATH']}"
    else:
        environ["PYTHONPATH"] = build_path
    return environ


def _ext_suffix() -> str:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("Unable to determine Python extension suffix")
    return ext_suffix


def _build_extension(
    *,
    makefile: Path,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir.joinpath(f"_C{_ext_suffix()}")
    if output_path.exists():
        output_path.unlink()

    cmd = ["make", "-f", str(makefile), f"OUT={output_path}", "CONFIG=pytorch"]
    subprocess.check_call(cmd, cwd=TK_TOOLS_PATH, env=_get_env())


def _prepare_package_layout():
    TK_PACKAGE_PATH.mkdir(parents=True, exist_ok=True)
    for relative_path, content in PACKAGE_SOURCES.items():
        target = TK_PACKAGE_PATH.joinpath(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def test_tk_attn_h100_fwd():
    cmd = [
        sys.executable,
        "-c",
        "import thunderkittens as tk; tk.mha_forward; tk.fp8_gemm; tk.bf16_b200_gemm",
    ]
    subprocess.check_call(cmd, cwd=REPO_PATH, env=_get_env())


def install_tk():
    _prepare_package_layout()
    _build_extension(
        makefile=TK_TOOLS_PATH.joinpath("mha_h100.Makefile"),
        output_dir=TK_PACKAGE_PATH.joinpath("mha_h100"),
    )
    _build_extension(
        makefile=TK_TOOLS_PATH.joinpath("fp8_h100.Makefile"),
        output_dir=TK_PACKAGE_PATH.joinpath("fp8_h100"),
    )
    _build_extension(
        makefile=TK_TOOLS_PATH.joinpath("bf16_b200_gemm.Makefile"),
        output_dir=TK_PACKAGE_PATH.joinpath("bf16_b200"),
    )
    test_tk_attn_h100_fwd()
