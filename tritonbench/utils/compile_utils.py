"""Per-device default backend for ``torch.compile``.

``torch_tpu`` autoloads on ``import torch`` and monkeypatches ``torch.compile``
to default to ``backend="tpu"`` process-globally. That breaks ``torch.compile``
on cpu/cuda tensors whenever torch_tpu is merely installed
(google-pytorch/torch_tpu#1912), and it never sets the ``dynamic=False`` that
the TPU backend requires.

``set_compile_backend_for_device`` wraps ``torch.compile`` so that calls which
do not pass an explicit ``backend=`` get a device-appropriate default:

- ``tpu``    -> ``backend="tpu"``, ``dynamic=False`` (Inductor-only ``mode`` /
  ``options`` kwargs are dropped, since the TPU backend rejects them)
- otherwise  -> ``backend="inductor"`` (undo torch_tpu's global default)

Calls that pass an explicit non-``None`` ``backend`` are left untouched (a
missing backend, or an explicit ``backend=None``, takes the device default).
This keeps the ~40 operator ``torch.compile`` call sites working per device
without editing them.
"""

import functools
import importlib.util

import torch

# The torch.compile we wrap (torch_tpu's patched one, or stock torch's),
# captured once so repeated installs don't stack wrappers.
_BASE_COMPILE = None


def _torch_tpu_installed() -> bool:
    return importlib.util.find_spec("torch_tpu") is not None


def set_compile_backend_for_device(device: str) -> None:
    """Install a device-appropriate default backend for ``torch.compile``.

    Only intervenes where torch_tpu's global default is actually wrong:
      - ``tpu``: force ``backend="tpu"`` (the whole point).
      - ``cpu``/``cuda`` **and** torch_tpu installed: force ``backend="inductor"``
        to undo torch_tpu's global override.
    Any other device (e.g. ``mtia``, which registers its own dynamo backend) is
    left untouched, as is the case where torch_tpu isn't installed. Idempotent.
    """
    if device == "tpu":
        force_backend = "tpu"
    elif device in ("cpu", "cuda") and _torch_tpu_installed():
        force_backend = "inductor"
    else:
        return

    # Idempotent: _run installs this per op, so skip if already wrapped for
    # this device (avoids re-wrapping on every call).
    if getattr(torch.compile, "_tb_compile_device", None) == device:
        return

    global _BASE_COMPILE
    if _BASE_COMPILE is None:
        _BASE_COMPILE = torch.compile
    base = _BASE_COMPILE

    @functools.wraps(base)
    def _compile(model=None, **kwargs):
        # ``backend`` is keyword-only in stock torch.compile (and we assume
        # torch_tpu's patched version keeps it so), so callers can only pass it
        # via kwargs; a missing or None backend means "use the default".
        if kwargs.get("backend") is None:
            kwargs["backend"] = force_backend
            if force_backend == "tpu":
                kwargs.setdefault("dynamic", False)
                # The TPU backend rejects Inductor-only knobs; drop them so
                # torch.compile baselines that request them still run on TPU.
                kwargs.pop("mode", None)
                kwargs.pop("options", None)
        return base(model, **kwargs)

    _compile._tb_compile_device = device  # marker for debugging/introspection
    torch.compile = _compile
