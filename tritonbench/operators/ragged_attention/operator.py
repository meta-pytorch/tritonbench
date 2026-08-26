import argparse
import contextlib
import os
from typing import Any, Callable, List, Optional

import torch
from tritonbench.utils.env_utils import (
    get_nvidia_gpu_model,
    IS_BLACKWELL,
    is_cuda,
    is_fbcode,
    is_hip_mi350,
)
from tritonbench.utils.input import input_filter
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
)

from .hstu import get_test_inputs, HAS_HAMMER, triton_hstu_mha, triton_ragged_hstu_mha
from .triton_autows import (
    triton_autows_ragged_hstu,
    triton_autows_ragged_hstu_persistent,
)


@contextlib.contextmanager
def _scoped_env(overrides: dict[str, str]):
    """Apply env overrides for the duration of the block, then restore them.

    The meta-autoWS compiler toggles (TRITON_USE_META_WS etc.) are read at
    compile time and are NOT part of the JIT cache key, so leaving them set leaks
    into whichever backend recompiles next in the input sweep -- e.g. the non-WS
    `hstu` baseline would silently recompile warp-specialized. Scoping them around
    each autoWS kernel keeps every backend's config independent of run order.
    """
    prev = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


HAS_CUDA = False
try:
    HAS_CUDA = is_fbcode() and is_cuda() and not IS_BLACKWELL
except (FileNotFoundError, AttributeError):
    HAS_CUDA = False

if HAS_CUDA:
    from .fb.hstu import cuda_hstu_mha

HAS_CUDA_BLACKWELL = False
try:
    HAS_CUDA_BLACKWELL = is_fbcode() and is_cuda() and IS_BLACKWELL
except (FileNotFoundError, AttributeError):
    HAS_CUDA_BLACKWELL = False

if HAS_CUDA_BLACKWELL:
    try:
        from generative_recommenders.fb.ultra.ops.blackwell.hstu_mha_blackwell import (
            hstu_mha_blackwell,
        )
    except ImportError:
        HAS_CUDA_BLACKWELL = False

if is_fbcode():
    from tritonbench.utils.fb.hstu_prod import get_prod_config
else:
    get_prod_config = lambda x: None

# HSTU self-attention kernels ported to MetaMain2 triton tutorials
# (third_party/tlx/tutorials/hstu_self_attn), imported like the blackwell FA
# tutorials. Multi-file port with bare intra-package imports, so add its
# directory to sys.path and import the two entrypoints by module name:
#   hstu_self_triton_mha - hammer-template Triton self-attn (fwd+bwd)
#   hstu_self_tlx_mha     - hammer-template TLX Blackwell self-attn (fwd+bwd)
HAS_HSTU_SELF_ATTN = False
try:
    import os as _os
    import sys as _sys

    import triton.language.extra.tlx.tutorials as _tlx_tut

    _hstu_self_dir = _os.path.join(list(_tlx_tut.__path__)[0], "hstu_self_attn")
    if _os.path.isdir(_hstu_self_dir):
        if _hstu_self_dir not in _sys.path:
            _sys.path.insert(0, _hstu_self_dir)
        from tlx_bw_hstu_attention import tlx_bw_hstu_mha as hstu_self_tlx_mha
        from triton_hstu_attention import (
            configure_autows as hstu_self_configure,
            triton_hstu_mha as hstu_self_triton_mha,
        )

        HAS_HSTU_SELF_ATTN = True
except Exception:
    HAS_HSTU_SELF_ATTN = False

# gfx950 (MI350X) TLX HSTU self-attention, from the same tutorials directory.
# Separate try block from the Blackwell import above so neither disables the
# other: only one of the two ever has usable hardware in a given run.
# Defined unconditionally: parse_op_args() reads them to build --tlx-gfx950-bwd-variant
# regardless of platform. The `isdir` check below can fail *without* raising -- the
# tutorials module imports fine under the fbcode triton while hstu_self_attn/ is not
# materialized on disk -- so relying on the `except` branch to define them left them
# unbound and NameError'd the whole operator on non-gfx950 CI.
HAS_HSTU_TLX_GFX950 = False
HSTU_GFX950_BWD_VARIANTS: dict = {}
HSTU_GFX950_DEFAULT_BWD_VARIANT: str = ""
try:
    import os as _os
    import sys as _sys

    import triton.language.extra.tlx.tutorials as _tlx_tut

    _hstu_self_dir = _os.path.join(list(_tlx_tut.__path__)[0], "hstu_self_attn")
    if _os.path.isdir(_hstu_self_dir):
        if _hstu_self_dir not in _sys.path:
            _sys.path.insert(0, _hstu_self_dir)
        from tlx_gfx950_ragged_hstu_attention import (
            BWD_VARIANTS as HSTU_GFX950_BWD_VARIANTS,
            DEFAULT_BWD_VARIANT as HSTU_GFX950_DEFAULT_BWD_VARIANT,
            tlx_gfx950_hstu_mha,
        )

        HAS_HSTU_TLX_GFX950 = True
except Exception:
    HAS_HSTU_TLX_GFX950 = False


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--min-seq-len-log2", type=int, default=8)
    parser.add_argument("--max-seq-len-log2", type=int, default=10)
    parser.add_argument("--seq-sparsity", type=float, default=1.0)
    parser.add_argument("--has-delta-q", type=bool, default=False)
    parser.add_argument("--delta-size", type=int, default=256)
    parser.add_argument("--target-size", type=int, default=20)
    parser.add_argument("--max-attn-len", type=int, default=0)
    # set to 0 to use hstu_mha
    parser.add_argument("--min-full-attn-seq-len", type=int, default=0)
    parser.add_argument("--contextual-seq-len", type=int, default=0)
    parser.add_argument("--sampling-alpha", type=float, default=1.7)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--attn-mask-type", type=str, default="lower_triangular")
    parser.add_argument(
        "--tlx-gfx950-bwd-variant",
        type=str,
        default=None,
        choices=sorted(HSTU_GFX950_BWD_VARIANTS) or None,
        help=(
            "Backward schedule for the hstu_tlx_gfx950 backend "
            f"(default: {HSTU_GFX950_DEFAULT_BWD_VARIANT or 'n/a'}). "
            "Forward is identical across variants."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config specifies a preset config. Most other args will be ignored.",
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        prod_config = get_prod_config(args.config)
        if prod_config:
            self.batch_size = prod_config.batch_size
            self.num_heads = prod_config.num_heads
            self.attn_dim = prod_config.attn_dim
            self.hidden_dim = prod_config.hidden_dim
            self.min_seq_len_log2 = prod_config.seq_len_log2
            self.max_seq_len_log2 = prod_config.seq_len_log2
            self.sparsity_seq = prod_config.sparsity_seq
            # TODO: support delta_q in prod config
            self.has_delta_q = False
            self.delta_size = 0
            self.target_size = prod_config.target_size
            self.max_attn_len = prod_config.max_attn_len
            # TODO: support min_full_attn_seq_len in prod config
            self.min_full_attn_seq_len = 0
            # TODO: support contextual_seq_len in prod config
            self.contextual_seq_len = 0
            self.alpha = (
                prod_config.alpha
                if prod_config.alpha is not None
                else 1.0 / self.attn_dim
            )
            self.attn_mask_type = prod_config.attn_mask_type
        else:
            self.batch_size = args.batch_size
            self.num_heads = args.heads
            self.attn_dim = args.attn_dim
            self.hidden_dim = args.hidden_dim
            self.min_seq_len_log2 = args.min_seq_len_log2
            self.max_seq_len_log2 = args.max_seq_len_log2
            self.sparsity_seq = [args.seq_sparsity]
            self.has_delta_q = args.has_delta_q
            self.delta_size = args.delta_size
            self.target_size = args.target_size
            self.max_attn_len = args.max_attn_len
            self.min_full_attn_seq_len = args.min_full_attn_seq_len
            self.contextual_seq_len = args.contextual_seq_len
            self.alpha = 1.0 / self.attn_dim
            self.attn_mask_type = args.attn_mask_type
        self.causal = args.causal
        self.sampling_alpha = args.sampling_alpha
        self.tlx_gfx950_bwd_variant = (
            args.tlx_gfx950_bwd_variant or HSTU_GFX950_DEFAULT_BWD_VARIANT
        )
        self.requires_grad = not (self.mode == Mode.FWD_NO_GRAD)

    @register_benchmark(enabled=is_cuda(), baseline=True)
    def hstu(self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity):
        # TMA is NVIDIA Hopper+ only; on AMD the backward kernel crashes when
        # tensor-descriptor rewrite and tl.assume (buffer-ops) coexist.
        _enable_tma = is_cuda()
        return lambda: triton_hstu_mha(
            max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=self.max_attn_len,
            contextual_seq_len=self.contextual_seq_len,
            sort_by_length=True,
            enable_tma=_enable_tma,
        )

    @register_benchmark(enabled=HAS_HSTU_SELF_ATTN and is_cuda())
    def hstu_triton_hammer(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        # Hammer-template Triton self-attn (MetaMain2 port). Needs an explicit
        # attn_scale tensor (the GR `hstu` baseline bakes 1/max_seq_len).
        # Reset to the plain (non-autoWS) config in case an autoWS backend below
        # switched it earlier in this process.
        hstu_self_configure(autows=False)
        attn_scale = torch.tensor(
            1.0 / max_seq_len, device=q.device, dtype=torch.float32
        )
        return lambda: hstu_self_triton_mha(
            max_seq_len=max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            num_targets=num_targets,
            max_attn_len=self.max_attn_len,
            contextual_seq_len=self.contextual_seq_len,
            sort_by_length=True,
            enable_tma=is_cuda(),
        )

    @register_benchmark(enabled=HAS_HSTU_SELF_ATTN and IS_BLACKWELL)
    def hstu_tlx(self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity):
        # Hammer-template TLX (Blackwell warp-specialized) self-attn. SiLU heads
        # only (num_softmax_heads=0); scalar attn_scale.
        attn_scale = torch.tensor(
            1.0 / max_seq_len, device=q.device, dtype=torch.float32
        )
        return lambda: hstu_self_tlx_mha(
            max_seq_len=max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            num_softmax_heads=0,
            num_targets=num_targets,
            max_attn_len=self.max_attn_len,
            contextual_seq_len=self.contextual_seq_len,
            causal=True,
        )

    def _hstu_tlx_gfx950(
        self, bwd_variant, q, k, v, seq_offsets, num_targets, max_seq_len
    ):
        # gfx950 TLX HSTU. `attn_scale=None` makes the kernel fold 1/max_seq_len
        # into the silu itself, which is what the `hstu` baseline does -- and the
        # FA-schedule backwards reject an explicit attn_scale outright.
        #
        # The FA-schedule variants additionally require the plain causal HSTU
        # path: num_targets set, no contextual prefix / max_attn_len /
        # full_attn_size, no length sorting, silu heads only, Dq=Dv=128. Run with
        # the operator defaults (--target-size 20 --attn-dim 128 --hidden-dim 128)
        # or pick `default` / `native_mfma_*` for the unconstrained backward.
        return lambda: tlx_gfx950_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=None,
            num_targets=num_targets,
            invalid_attn_mask_type=self.attn_mask_type,
            max_attn_len=self.max_attn_len,
            contextual_seq_len=self.contextual_seq_len,
            full_attn_size=0,
            sort_by_length=False,
            num_softmax_heads=0,
            bwd_variant=bwd_variant,
        )

    @register_benchmark(
        enabled=HAS_HSTU_TLX_GFX950 and is_hip_mi350(),
        tags=["tlx", "amd", "gfx950"],
    )
    def hstu_tlx_gfx950(self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity):
        # Backward schedule from --tlx-gfx950-bwd-variant; defaults to the
        # BN128 resident-K mask-peel variant, which is the fastest at N >= 2048.
        return self._hstu_tlx_gfx950(
            self.tlx_gfx950_bwd_variant,
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            max_seq_len,
        )

    @register_benchmark(
        enabled=HAS_HSTU_TLX_GFX950 and is_hip_mi350(),
        tags=["tlx", "amd", "gfx950"],
    )
    def hstu_tlx_gfx950_bn256(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        # BN256 direct-load FA schedule: still the best backward at N == 1024.
        return self._hstu_tlx_gfx950(
            "kv_parallel_fa_schedule_bn256_direct_qdo_g2l",
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            max_seq_len,
        )

    @register_benchmark(
        enabled=HAS_HSTU_TLX_GFX950 and is_hip_mi350(),
        tags=["tlx", "amd", "gfx950"],
    )
    def hstu_tlx_gfx950_ordinary_dot(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        # Ordinary-dot dQ backward with the kernel's own kv_parallel heuristic.
        # The one variant with no FA-schedule preconditions, so it is the
        # backend to use for masked / contextual / softmax-head shapes.
        return self._hstu_tlx_gfx950(
            "default", q, k, v, seq_offsets, num_targets, max_seq_len
        )

    def _hstu_self_autows(
        self, cfg, q, k, v, seq_offsets, num_targets, max_seq_len, smem_search=False
    ):
        # Meta-autoWS self-attn. The structural config (autows/dp/manual_dp/...) is
        # switched in-process via configure_autows() (rebuilds autotune configs +
        # clears the JIT caches / used-global guard), so multiple autoWS variants
        # can be benchmarked in one process. The compiler WS toggles are env vars
        # scoped to the returned callable so they do not leak into other backends
        # (e.g. the non-WS `hstu` baseline) that recompile later in the sweep.
        ws_env = {
            "TRITON_USE_META_WS": "1",
            "TRITON_DISABLE_WSBARRIER_REORDER": "1",
        }
        if smem_search:
            ws_env["TRITON_WS_SMEM_PLAN_SEARCH"] = "1"
        with _scoped_env(ws_env):
            hstu_self_configure(**cfg)
        attn_scale = torch.tensor(
            1.0 / max_seq_len, device=q.device, dtype=torch.float32
        )

        def _run():
            with _scoped_env(ws_env):
                return hstu_self_triton_mha(
                    max_seq_len=max_seq_len,
                    alpha=self.alpha,
                    q=q,
                    k=k,
                    v=v,
                    seq_offsets=seq_offsets,
                    attn_scale=attn_scale,
                    num_targets=num_targets,
                    max_attn_len=self.max_attn_len,
                    contextual_seq_len=self.contextual_seq_len,
                    sort_by_length=True,
                    enable_tma=is_cuda(),
                )

        return _run

    # fwd_only: the fwd-data-partition variants share the same (RMW) WS backward,
    # and two different WS backward configs cannot be compiled/run in one process
    # (sequential meta-WS bwd launches deadlock), so only hstu_triton_autows_dqreduce
    # exercises the WS backward -- these variants benchmark the forward only. That
    # backend is currently disabled (see below), so no backend covers the WS bwd.
    @register_benchmark(enabled=HAS_HSTU_SELF_ATTN and IS_BLACKWELL, fwd_only=True)
    def hstu_triton_autows(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        # Default meta-autoWS: warp-specialized KV loop, DP=1 (fwd only).
        return self._hstu_self_autows(
            dict(autows=True, dp=1, pin=True),
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            max_seq_len,
        )

    @register_benchmark(enabled=HAS_HSTU_SELF_ATTN and IS_BLACKWELL, fwd_only=True)
    def hstu_triton_autows_manualdp(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        # Manual fwd data-partition (split BLOCK_M, shared K/V, 2 MMA groups).
        return self._hstu_self_autows(
            dict(autows=True, manual_dp=True, dp=2, warps=4, pin=True),
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            max_seq_len,
        )

    @register_benchmark(enabled=False, fwd_only=True)
    def hstu_triton_autows_dp2(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        # Compiler data_partition_factor=2 fwd.
        return self._hstu_self_autows(
            dict(autows=True, dp=2, warps=4, pin=True),
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            max_seq_len,
        )

    # TODO: Re-enable (fwd and bwd) once the backward stops aborting the Triton
    # compiler. Disabled in both modes today for this reason:
    # `_scoped_env` only wraps the forward call, so `_hstu_attn_bwd` -- compiled
    # later by autograd, outside that scope -- runs through the *upstream* autoWS
    # pipeline (`use-meta-ws=false`) on a `tt.warp_specialize` loop and trips
    # `InsertAref.cpp: assert(consumers.size() > 0)` in NVWSInsertAref, surfaced as
    # `RuntimeError: PassManager::run failed`. Holding the WS env across the
    # backward does compile and run it, so the fix is on the caller side, plus a
    # compiler that rejects WS instead of asserting -- there is no non-WS fallback
    # for this config either (with WS off these bwd tiles, BM=BN=128 / ns=2 /
    # dq_reuse, need 640 TMEM columns against a 512 limit).
    # The forward compiles and runs fine on its own; it is disabled along with the
    # backward rather than demoted to fwd_only because this is the only backend
    # covering the WS backward and the whole variant is what needs to come back.
    @register_benchmark(enabled=HAS_HSTU_SELF_ATTN and IS_BLACKWELL and False)
    def hstu_triton_autows_dqreduce(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        # TLX-matching dq-reduce bwd: BM=BN=128, ns=2, TMEM reuse, dq via TMA reduce.
        return self._hstu_self_autows(
            dict(
                autows=True,
                dq_reduce=True,
                dq_reuse=True,
                dp=1,
                bwd_bm=128,
                bwd_bn=128,
                bwd_stages=2,
                warps=4,
                dq_iters=4,
                pin=True,
            ),
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            max_seq_len,
            smem_search=True,
        )

    @register_benchmark(enabled=HAS_HAMMER)
    def hammer_hstu(self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity):
        return lambda: triton_ragged_hstu_mha(
            N=max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            invalid_attn_mask_type=self.attn_mask_type,
            num_targets=num_targets,
            attn_scale=None,
            attn_bias=None,
            seq2_offsets=None,
            max_attn_len=self.max_attn_len,
            contextual_seq_len=self.contextual_seq_len,
            sort_by_length=False,
            full_attn_size=0,
        )

    # Test with --force. This backend currently does not compile: Meta AutoWS
    # crashes in NVGPUWarpSpecialization for the HSTU SiLU product consumed by
    # the PV MMA across partition boundaries.
    # TODO: Enable once the compiler issue is fixed and runtime gates for Meta
    # Triton + Blackwell + AutoWS are validated.
    @register_benchmark(enabled=False, fwd_only=True)
    def triton_autows_ragged_hstu(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        return lambda: triton_autows_ragged_hstu(
            max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
        )

    # Persistent AutoWS variant (K-6, D109223946): warp-specializes an outer
    # persistent tile loop (bounded by each jagged sequence's valid M-tile count)
    # and handles num_targets clamping; validated ws=pass. Disabled by default
    # for the same runtime-gate reason; test with `--force` under
    # TRITON_USE_META_WS=1.
    @register_benchmark(enabled=False, fwd_only=True)
    def triton_autows_ragged_hstu_persistent(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        return lambda: triton_autows_ragged_hstu_persistent(
            max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
        )

    # TODO: remove B200 hacks like these.
    @register_benchmark(enabled=(HAS_CUDA))
    def hstu_cuda(self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity):
        return lambda: cuda_hstu_mha(
            max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=self.causal,
            num_targets=num_targets,
            max_attn_len=self.max_attn_len,
            min_full_attn_seq_len=self.min_full_attn_seq_len,
            contextual_seq_len=self.contextual_seq_len,
            sort_by_length=True,
        )

    @register_benchmark(enabled=HAS_CUDA_BLACKWELL)
    def hstu_cuda_blackwell(
        self, q, k, v, seq_offsets, num_targets, max_seq_len, sparsity
    ):
        return lambda: hstu_mha_blackwell(
            max_seq_len=max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=self.causal,
            num_targets=num_targets,
            max_attn_len=self.max_attn_len,
            min_full_attn_seq_len=self.min_full_attn_seq_len,
            contextual_seq_len=self.contextual_seq_len,
            sort_by_length=True,
        )

    def get_x_val(self, example_inputs):
        seq_len = example_inputs[-2]
        sparsity = example_inputs[-1]
        return (
            self.batch_size,
            self.num_heads,
            seq_len,
            self.attn_dim,
            self.hidden_dim,
            sparsity,
            self.target_size,
            self.max_attn_len,
        )

    def get_available_num_inputs(self) -> int:
        return ((self.max_seq_len_log2 + 1) - self.min_seq_len_log2) * len(
            self.sparsity_seq
        )

    def get_input_iter(self):
        for sparsity in self.sparsity_seq:
            for seq_len in [
                2**i for i in range(self.min_seq_len_log2, self.max_seq_len_log2 + 1)
            ]:
                yield get_test_inputs(
                    self.batch_size,
                    self.num_heads,
                    seq_len,
                    self.attn_dim,
                    self.hidden_dim,
                    sparsity,
                    self.has_delta_q,
                    self.delta_size,
                    self.target_size,
                    self.max_attn_len,
                    self.dtype,
                    requires_grad=self.requires_grad,
                )

    def _flops(
        self,
        batch_size: int,
        max_seqlen: int,
        attn_dim: int,
        hidden_dim: int,
        nheads: int,
        seq_offsets: torch.Tensor,
        mode: str = "fwd",
    ) -> float:
        assert mode in ["fwd", "bwd", "fwd_bwd"]
        ratio = 2.0  # triangular masking
        f1 = 0.0
        f2 = 0.0
        for i in range(batch_size):
            seq_len = int((seq_offsets[i + 1] - seq_offsets[i]).item())
            # (QK^T), dQ = d(QK^T)K, dK^T = Q^Td(QK^T)
            f1 += 2 * nheads * attn_dim * seq_len**2 // ratio
            # (QK^T)V, d(QK^T) = dOV^T, dV = (QK^T)^TdO,
            f2 += 2 * nheads * hidden_dim * seq_len**2 // ratio
        if mode == "fwd":
            return f1 + f2  # computes (QK^T) and (QK^T)V
        elif mode == "bwd":
            return 3 * f1 + 2 * f2  # computes (QK^T), dQ, dK, dV, d(QK^T)
        else:
            return 4 * f1 + 3 * f2

    @register_metric()
    def flops(
        self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q, k, v, seq_offsets, num_targets, max_seq_len, _ = example_inputs
        flops = self._flops(
            self.batch_size,
            max_seq_len,
            self.attn_dim,
            self.hidden_dim,
            self.num_heads,
            seq_offsets,
            mode=self.mode.value,
        )
        return flops
