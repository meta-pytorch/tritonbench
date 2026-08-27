# utils to identify triton versions

import functools
import importlib.util
from typing import Any, Dict

import triton.language as tl


def has_warp_spec():
    import triton.language as tl

    return hasattr(tl, "async_task")


def has_new_tma():
    import triton
    import triton.language as tl

    # Check basic TMA API availability
    if not (hasattr(triton, "set_allocator") and hasattr(tl, "make_tensor_descriptor")):
        return False

    return True


@functools.lru_cache
def has_tlx():
    """
    Returns whether TLX is supported.
    """
    # TODO: Replace with the variant in compat once that's
    # available in OSS.
    tlx_module = "triton.language.extra.tlx"
    spec = importlib.util.find_spec(tlx_module)
    return spec is not None


@functools.lru_cache
def has_torch_tlx():
    """
    Returns whether the installed torch exposes the inductor `triton.tlx_mode`
    knob -- the torch-side half of torchTLX. Providers that run TLX templates
    through torch.compile (torch_tlx_* benchmarks) need this in addition to
    has_tlx(): has_tlx() only says the Triton TLX module exists, not that torch's
    inductor config can select it. Older torch builds omit the knob and would
    otherwise raise AttributeError inside inductor_config.patch(...).
    """
    try:
        import torch._inductor.config as inductor_config

        return hasattr(inductor_config.triton, "tlx_mode")
    except (ImportError, AttributeError):
        return False


#: Tag shared by the whole torchTLX perf comparison set for an operator: the
#: `torch_tlx_<op>` provider plus the `pt2_*` baseline it is measured against.
#: `--tags torch_tlx` therefore selects a complete comparison in one flag, and
#: stays correct as new TLX providers are added.
TORCH_TLX_TAG = "torch_tlx"

#: Full tag set for a `torch_tlx_<op>` provider. Operators append their own
#: hardware tags (e.g. "amd", "gfx950") matching the provider's `enabled` gate.
TORCH_TLX_TAGS = ["tlx", TORCH_TLX_TAG, "pt2"]


def torch_tlx_inductor_config(**overrides: Any) -> Dict[str, Any]:
    """Inductor config patch shared by every `torch_tlx_<op>` provider.

    PT2 max-autotune with TLX "allow" mode, so TLX templates compete against the
    standard Triton templates during autotuning. Identical to the matching `pt2_*`
    baseline except for `triton.tlx_mode` -- that single difference is what makes
    the reported speedup a clean PT2-vs-PT2+TLX comparison. `force_disable_caches`
    forces a real recompile so TLX candidates are not served from the baseline's
    autotune cache.

    Perf benchmarks always use "allow"; "force" mode is reserved for correctness
    tests, where the point is to exercise the TLX template rather than to let it
    win autotuning on merit.
    """
    config: Dict[str, Any] = {
        "max_autotune": True,
        "autotune_fallback_to_aten": False,
        "force_disable_caches": True,
        "triton.tlx_mode": "allow",
    }
    config.update(overrides)
    return config


def has_experimental_descriptor():
    import triton.language as tl

    return hasattr(getattr(tl, "tools", None), "experimental_descriptor")
