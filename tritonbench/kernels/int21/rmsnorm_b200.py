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


def _ensure_last_dim_contiguous(t: Tensor) -> Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_tensor(name: str, tensor: Tensor, x: Tensor, shape=None) -> None:
    _require(tensor.is_cuda, f"{name} must be a CUDA tensor")
    _require(tensor.device == x.device, f"{name} must be on {x.device}")
    _require(
        tensor.dtype in _SUPPORTED_DTYPES, f"unsupported {name} dtype: {tensor.dtype}"
    )
    if shape is not None:
        _require(tensor.shape == shape, f"{name} shape must be {tuple(shape)}")
    _require(tensor.stride(-1) == 1, f"{name} last dimension must be contiguous")


def _check_affine(name: str, tensor: Tensor, x: Tensor) -> None:
    _check_tensor(name, tensor, x)
    heads = x.size(1) if x.dim() == 3 else 1
    expected = {(x.size(-1),), (heads, x.size(-1))}
    _require(
        tuple(tensor.shape) in expected,
        f"{name} shape must be one of {sorted(expected)}",
    )


@torch.library.custom_op(
    "int21::rmsnorm_fwd_out",
    mutates_args=("out", "residual_out", "rstd"),
    device_types="cuda",
    schema="(Tensor x, Tensor weight, Tensor bias, Tensor residual, Tensor(a!) out, Tensor(b!) residual_out, Tensor(c!) rstd, float eps) -> ()",
)
def _rmsnorm_fwd_out(
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


@_rmsnorm_fwd_out.register_fake
def _rmsnorm_fwd_out_fake(*args, **kwargs):
    return None


@torch.library.custom_op(
    "int21::rmsnorm_bwd_out",
    mutates_args=("dx", "dw", "db", "dresidual"),
    device_types="cuda",
    schema="(Tensor x, Tensor weight, Tensor dout, Tensor rstd, Tensor dresidual_out, Tensor(a!) dx, Tensor(b!) dw, Tensor(c!) db, Tensor(d!) dresidual) -> ()",
)
def _rmsnorm_bwd_out(
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


@_rmsnorm_bwd_out.register_fake
def _rmsnorm_bwd_out_fake(*args, **kwargs):
    return None


def _call_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    residual: Tensor,
    out: Tensor,
    residual_out: Tensor,
    rstd: Tensor,
    eps: float,
) -> None:
    if torch.compiler.is_compiling():
        _rmsnorm_fwd_out(x, weight, bias, residual, out, residual_out, rstd, float(eps))
    else:
        ext = _EXT if _EXT is not None else load_ext()
        ext.rmsnorm_fwd(x, weight, bias, residual, out, residual_out, rstd, float(eps))


def _call_bwd(
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
    if torch.compiler.is_compiling():
        _rmsnorm_bwd_out(x, weight, dout, rstd, dresidual_out, dx, dw, db, dresidual)
    else:
        ext = _EXT if _EXT is not None else load_ext()
        ext.rmsnorm_bwd(x, weight, dout, rstd, dresidual_out, dx, dw, db, dresidual)


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    _require(x.is_cuda, "x must be a CUDA tensor")
    _require(x.dim() in (2, 3), "x must be 2D or 3D after flattening")
    _require(x.size(-1) > 0, "x last dimension must be positive")
    _require(x.dtype in _SUPPORTED_DTYPES, "unsupported x dtype")
    _require(x.stride(-1) == 1, "x last dimension must be contiguous")
    if weight is not None:
        _check_affine("weight", weight, x)
    if bias is not None:
        _check_affine("bias", bias, x)
    if residual is not None:
        _check_tensor("residual", residual, x, x.shape)
    if out_dtype is not None:
        _require(
            out_dtype in _SUPPORTED_DTYPES, f"unsupported output dtype: {out_dtype}"
        )
    if residual_dtype is not None:
        _require(
            residual_dtype in _SUPPORTED_DTYPES,
            f"unsupported residual dtype: {residual_dtype}",
        )

    out_dtype = x.dtype if out_dtype is None else out_dtype
    out = torch.empty_like(x, dtype=out_dtype)
    rstd = (
        torch.empty(*x.shape[:-1], device=x.device, dtype=torch.float32)
        if store_rstd
        else None
    )

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
        _call_fwd(
            x,
            weight if weight is not None else _empty_arg(x, torch.float32),
            bias if bias is not None else _empty_arg(x, torch.float32),
            residual if residual is not None else _empty_arg(x),
            out,
            residual_out if residual_out is not None else _empty_arg(x),
            rstd if rstd is not None else _empty_arg(x, torch.float32),
            eps,
        )

    return out, (x if residual_out is None else residual_out), rstd


def rmsnorm_bwd(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dresidual_out: Optional[Tensor] = None,
    has_bias: bool = False,
    has_residual: bool = False,
    bias: Optional[Tensor] = None,
):
    _require(x.is_cuda, "x must be a CUDA tensor")
    _require(x.dim() in (2, 3), "x must be 2D or 3D after flattening")
    _require(x.size(-1) > 0, "x last dimension must be positive")
    _require(x.dtype in _SUPPORTED_DTYPES, "unsupported x dtype")
    _require(x.stride(-1) == 1, "x last dimension must be contiguous")
    _check_tensor("dout", dout, x, x.shape)
    _require(rstd.is_cuda and rstd.device == x.device, f"rstd must be on {x.device}")
    _require(rstd.dtype == torch.float32, "rstd must be float32")
    _require(rstd.shape == x.shape[:-1], f"rstd shape must be {tuple(x.shape[:-1])}")
    if dresidual_out is not None:
        _check_tensor("dresidual_out", dresidual_out, x, x.shape)
    dx = torch.empty_like(x)
    dresidual = (
        torch.empty_like(x, dtype=dresidual_out.dtype)
        if dresidual_out is not None and dresidual_out.dtype != dx.dtype
        else None
    )

    N = x.size(-1)
    heads = x.size(1) if x.dim() == 3 else 1
    affine_shape = (heads, N) if x.dim() == 3 else (N,)

    if weight is not None:
        _check_affine("weight", weight, x)
        _require(
            weight.shape == affine_shape, "direct 3D backward requires per-head weight"
        )
        dw_accum = torch.empty(weight.shape, device=x.device, dtype=torch.float32)
    else:
        dw_accum = None
    if has_bias:
        bias_ref = bias if bias is not None else weight
        _require(bias_ref is not None, "bias gradient path needs bias or weight shape")
        _check_affine("bias", bias_ref, x)
        _require(
            bias_ref.shape == affine_shape, "direct 3D backward requires per-head bias"
        )
        db_accum = torch.empty(bias_ref.shape, device=x.device, dtype=torch.float32)
    else:
        db_accum = None

    if x.numel() == 0:
        dw = torch.zeros_like(weight) if weight is not None else None
        if has_bias:
            bias_ref = bias if bias is not None else weight
            db = torch.zeros_like(bias_ref)
        else:
            db = None
        if has_residual and dresidual is None:
            dresidual = dx
        return dx, dw, db, dresidual

    _call_bwd(
        x,
        weight if weight is not None else _empty_arg(x, torch.float32),
        dout,
        rstd,
        dresidual_out if dresidual_out is not None else _empty_arg(x),
        dx,
        dw_accum if dw_accum is not None else _empty_arg(x, torch.float32),
        db_accum if db_accum is not None else _empty_arg(x, torch.float32),
        dresidual if dresidual is not None else _empty_arg(x),
    )

    dw = dw_accum.to(weight.dtype) if weight is not None else None
    if has_bias:
        bias_dtype = bias.dtype if bias is not None else weight.dtype
        db = db_accum.to(bias_dtype)
    else:
        db = None
    if has_residual and dresidual is None:
        dresidual = dx
    return dx, dw, db, dresidual


class RMSNormPtxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Optional[Tensor],
        bias: Optional[Tensor] = None,
        residual: Optional[Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        residual_dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        prenorm: bool = False,
    ):
        x = _ensure_last_dim_contiguous(x)
        if residual is not None:
            residual = _ensure_last_dim_contiguous(residual)
        need_grad = any(ctx.needs_input_grad[:4])
        out, residual_out, rstd = rmsnorm_fwd(
            x,
            weight,
            bias=bias,
            residual=residual,
            out_dtype=out_dtype,
            residual_dtype=residual_dtype,
            eps=eps,
            store_rstd=need_grad,
        )
        ctx.save_for_backward(
            x if residual is None else residual_out, weight, bias, rstd
        )
        ctx.has_bias = bias is not None
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        if residual is not None and prenorm:
            return out, residual_out
        return out

    @staticmethod
    def backward(ctx, dout: Tensor, *args):
        x, weight, bias, rstd = ctx.saved_tensors
        dout = _ensure_last_dim_contiguous(dout)
        if ctx.prenorm and ctx.has_residual:
            dresidual_out = _ensure_last_dim_contiguous(args[0])
        else:
            dresidual_out = None
        dx, dw, db, dresidual = rmsnorm_bwd(
            x,
            weight,
            dout,
            rstd,
            dresidual_out=dresidual_out,
            has_bias=ctx.has_bias,
            has_residual=ctx.has_residual,
            bias=bias,
        )
        return dx, dw, db, dresidual, None, None, None, None


def rms_norm(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    prenorm: bool = False,
):
    """Autograd-aware RMSNorm using the int21 Blackwell (sm_100a) CUDA kernel.

    Backward is handled by :class:`RMSNormPtxFunction`. Returns the normalized
    output, or ``(out, residual_out)`` when ``residual`` is given and
    ``prenorm`` is set.
    """
    x_shape = x.shape
    per_head = (weight is not None and weight.dim() == 2) or (
        bias is not None and bias.dim() == 2
    )
    last_shape = x_shape[-2:] if per_head else x_shape[-1:]
    x_flat = x.reshape(-1, *last_shape)
    residual_flat = residual.reshape(-1, *last_shape) if residual is not None else None
    result = RMSNormPtxFunction.apply(
        x_flat, weight, bias, residual_flat, out_dtype, residual_dtype, eps, prenorm
    )
    if isinstance(result, tuple):
        return tuple(t.reshape(x_shape) for t in result)
    return result.reshape(x_shape)
