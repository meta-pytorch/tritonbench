# Adapted from INT21 AI's RMSNorm-B200 (https://github.com/Int21-AI/RMSNorm-B200)
# Original Copyright 2026 INT21 AI
# SPDX-License-Identifier: MIT
#
# Loads the bundled ``rmsnorm_b200.cu`` Blackwell (sm_100a) kernel via PyTorch's
# JIT cpp_extension loader and exports it as a PyTorch custom operator so it can
# be captured by ``torch.compile`` / Dynamo full-graph tracing.

import hashlib
import os
import site
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.cpp_extension import load


_EXT = None
_EMPTY_ARGS = {}
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _extra_include_paths():
    try:
        import pybind11
    except ImportError:
        return []
    return [pybind11.get_include()]


def _cuda_lib64() -> Optional[Path]:
    for root in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda-13.2",
        "/usr/local/cuda",
    ):
        if not root:
            continue
        lib64 = Path(root) / "lib64"
        if lib64.exists():
            return lib64
    return None


def load_ext():
    """Build (if needed) and load the int21 RMSNorm CUDA extension."""
    global _EXT
    if _EXT is None:
        here = Path(__file__).resolve().parent
        # REPO_ROOT/build/int21/ by default (here = .../tritonbench/kernels/int21)
        repo_root = Path(__file__).resolve().parents[3]
        build_root = Path(
            os.environ.get("INT21_BUILD_DIR", repo_root / "build" / "int21")
        )
        source = here / "rmsnorm_b200.cu"
        source_key = hashlib.sha256(
            str(here).encode() + source.read_bytes()
        ).hexdigest()[:12]
        build_dir = build_root / f"cuda_ext_{source_key}"
        build_dir.mkdir(parents=True, exist_ok=True)
        user_bin = str(Path(site.USER_BASE) / "bin")
        if user_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = user_bin + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
        cuda_lib64 = _cuda_lib64()
        extra_ldflags = [f"-Wl,-rpath,{cuda_lib64}"] if cuda_lib64 is not None else []
        _EXT = load(
            name=f"int21_rmsnorm_b200_ext_{source_key}",
            sources=[str(source)],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
            ],
            extra_include_paths=_extra_include_paths(),
            extra_ldflags=extra_ldflags,
            verbose=bool(int(os.environ.get("INT21_VERBOSE_BUILD", "0"))),
        )
    return _EXT


def _empty_arg(x: Tensor, dtype: Optional[torch.dtype] = None) -> Tensor:
    key = (x.device, dtype or x.dtype)
    empty = _EMPTY_ARGS.get(key)
    if empty is None:
        empty = torch.empty(0, device=key[0], dtype=key[1])
        _EMPTY_ARGS[key] = empty
    return empty


@torch.library.custom_op(
    "int21::rmsnorm_fwd_out",
    mutates_args=("out", "residual_out", "rstd"),
    device_types="cuda",
    schema="(Tensor x, Tensor weight, Tensor bias, Tensor residual, Tensor(a!) out, Tensor(b!) residual_out, Tensor(c!) rstd, float eps) -> ()",
)
def rmsnorm_fwd_out(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    residual: Tensor,
    out: Tensor,
    residual_out: Tensor,
    rstd: Tensor,
    eps: float,
) -> None:
    load_ext().rmsnorm_fwd(
        x, weight, bias, residual, out, residual_out, rstd, float(eps)
    )


@rmsnorm_fwd_out.register_fake
def _rmsnorm_fwd_out_fake(*args, **kwargs):
    return None


@torch.library.custom_op(
    "int21::rmsnorm_bwd_out",
    mutates_args=("dx", "dw", "db", "dresidual"),
    device_types="cuda",
    schema="(Tensor x, Tensor weight, Tensor dout, Tensor rstd, Tensor dresidual_out, Tensor(a!) dx, Tensor(b!) dw, Tensor(c!) db, Tensor(d!) dresidual) -> ()",
)
def rmsnorm_bwd_out(
    x: Tensor,
    weight: Tensor,
    dout: Tensor,
    rstd: Tensor,
    dresidual_out: Tensor,
    dx: Tensor,
    dw: Tensor,
    db: Tensor,
    dresidual: Tensor,
) -> None:
    load_ext().rmsnorm_bwd(x, weight, dout, rstd, dresidual_out, dx, dw, db, dresidual)


@rmsnorm_bwd_out.register_fake
def _rmsnorm_bwd_out_fake(*args, **kwargs):
    return None


def rmsnorm(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Forward RMSNorm using the int21 Blackwell CUDA kernel.

    Returns ``(out, residual_out, rstd)``. ``residual_out`` is ``None`` when no
    residual fusion is requested and ``rstd`` is always returned for downstream
    backward use.
    """
    if x.dim() not in (2, 3):
        raise AssertionError("x must be 2D or 3D after flattening")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise AssertionError(f"unsupported x dtype: {x.dtype}")

    out_dtype = x.dtype if out_dtype is None else out_dtype
    out = torch.empty_like(x, dtype=out_dtype)
    rstd = torch.empty(*x.shape[:-1], device=x.device, dtype=torch.float32)

    if residual is not None and residual_dtype is None:
        residual_dtype = residual.dtype
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = torch.empty_like(
            x, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
    else:
        residual_out = None

    if x.numel() != 0:
        rmsnorm_fwd_out(
            x,
            weight if weight is not None else _empty_arg(x, torch.float32),
            bias if bias is not None else _empty_arg(x, torch.float32),
            residual if residual is not None else _empty_arg(x),
            out,
            residual_out if residual_out is not None else _empty_arg(x),
            rstd,
            eps,
        )

    return out, residual_out, rstd
