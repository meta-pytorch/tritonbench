"""Generic Helion backend override for tritonbench.

This lets you experiment with a non-default Helion code-generation backend
(e.g. the CuTe DSL backend) for *any* operator's ``helion`` benchmark without
editing the operator or kernel definitions.

It works by:
  1. Setting ``HELION_BACKEND`` so every ``helion.kernel(...)`` that does not
     pass an explicit ``backend`` picks up the override (and so subprocess
     children inherit it).
  2. Wrapping ``helion.kernel`` to drop any hardcoded ``config=``/``configs=``
     arguments when the override is non-Triton. Those configs are Triton
     specific (indexing / pid_type / num_warps / ...) and are not valid for
     other backends; dropping them makes the kernel fall back to the chosen
     backend's own default config.

Usage: pass ``--helion-backend <name>`` on the tritonbench CLI, e.g.

    --op <any_op_with_a_helion_backend> --only helion --helion-backend cute
"""

import functools
import logging
import os

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_helion_backend_override(backend: str | None) -> None:
    """Route all Helion kernels in this process to ``backend``.

    No-op when ``backend`` is falsy or ``"triton"`` (the default). Idempotent.
    """
    global _PATCHED
    if not backend:
        return

    # Selection via env var: covers kernels that don't pass an explicit backend,
    # and is inherited by subprocess children (multi-device / in-task modes).
    os.environ["HELION_BACKEND"] = backend

    if backend == "triton" or _PATCHED:
        return

    try:
        import helion
    except ImportError:
        logger.warning("helion is not available; --helion-backend has no effect")
        return

    orig_kernel = helion.kernel

    @functools.wraps(orig_kernel)
    def _kernel(fn=None, *, config=None, configs=None, key=None, **settings):
        # Drop hardcoded Triton config(s); use the override backend's default.
        # Only set backend when the caller used keyword settings (not a Settings
        # object) and didn't already pin one — the env var covers the rest.
        if "settings" not in settings:
            settings.setdefault("backend", backend)
        return orig_kernel(fn, key=key, **settings)

    helion.kernel = _kernel
    # Keep the `jit` alias consistent if present.
    if getattr(helion, "jit", None) is orig_kernel:
        helion.jit = _kernel

    _PATCHED = True
    logger.warning(
        "Helion backend overridden to %r for all helion kernels "
        "(hardcoded config=/configs= are ignored; default config used).",
        backend,
    )
