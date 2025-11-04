# utils to identify triton versions

import functools
import importlib.util

import triton.language as tl


def has_warp_spec():
    import triton.language as tl

    return hasattr(tl, "async_task")


def has_new_tma():
    import triton
    import triton.language as tl

    return hasattr(triton, "set_allocator") and hasattr(tl, "make_tensor_descriptor")


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


def has_experimental_descriptor():
    import triton.language as tl

    return hasattr(getattr(tl, "tools", None), "experimental_descriptor")
