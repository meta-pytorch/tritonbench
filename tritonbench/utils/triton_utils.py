# utils to identify triton versions

import functools
import importlib.util

import triton.language as tl


class AsyncTaskContext:
    """Context manager that dispatches to tl.async_task if available, otherwise no-op."""

    def __init__(self, task_ids):
        self.task_ids = task_ids
        self._has_async_task = hasattr(tl, "async_task")
        self._context = None

    def __enter__(self):
        if self._has_async_task:
            self._context = tl.async_task(self.task_ids)
            return self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._has_async_task and self._context is not None:
            return self._context.__exit__(exc_type, exc_val, exc_tb)
        return False


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
