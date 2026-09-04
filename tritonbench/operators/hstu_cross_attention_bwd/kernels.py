# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Import shim for the HSTU cross-attention backward kernel.

The kernel lives in the Triton tree at
``third_party/tlx/tutorials/hstu_cross_attn/triton_bw_cross_attention.py`` and is
exposed through the installed Triton package as
``triton.language.extra.tlx.tutorials.hstu_cross_attn`` -- in an OSS checkout via
a symlink, in a buck build via
``fbsource//third-party/triton/beta/triton:tlx-hstu-cross-attn-tutorial``.
It is written as a standalone script rather than a package: it uses
``sys.path``-relative imports (``import stubs``, ``triton_attention_utils``) and
gates ``_HSTU_COMPUTE_FOLD`` on an env var read at *import* time. This shim sets
that env and puts the kernel directory on ``sys.path`` before importing, so the
operator can just ``from .kernels import xa, BwdVariant``.
"""

import importlib.util
import os
import sys

# The 2-KV data-partition variants (tlx_2kv / autows_2kv) require compute-fold,
# which the kernel reads as a module-level tl.constexpr at import time -- so it
# must be set BEFORE the import below. TRITON_ALLOW_NON_CONSTEXPR_GLOBALS lets the
# module use its module-level (non-constexpr) global config objects.
os.environ.setdefault("HSTU_COMPUTE_FOLD", "1")
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")


_KERNEL_PACKAGE = "triton.language.extra.tlx.tutorials.hstu_cross_attn"


def _kernel_dir():
    """Resolve the on-disk kernel directory, or None if it is not installed.

    Resolved as a module rather than by joining onto ``triton.__file__``: the OSS
    symlink and the buck link-tree put the directory in different places on disk
    even though both expose it at the same module path.
    """
    try:
        spec = importlib.util.find_spec(_KERNEL_PACKAGE)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin and os.path.basename(spec.origin) == "__init__.py":
        return os.path.dirname(spec.origin)
    # Namespace package (no __init__.py): the directory itself is the location.
    for location in spec.submodule_search_locations or ():
        return location
    return None


def _load():
    kernel_dir = _kernel_dir()
    if kernel_dir is None or not os.path.isdir(kernel_dir):
        raise ImportError(
            f"HSTU cross-attention kernels not found: `{_KERNEL_PACKAGE}` is not "
            "importable. This operator requires a Triton build that ships the tlx "
            "tutorials (buck: depend on "
            "fbsource//third-party/triton/beta/triton:tlx-hstu-cross-attn-tutorial)."
        )
    # The kernel module resolves `stubs` / `triton_attention_utils` relative to
    # its own directory, so it must be importable from sys.path.
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    import triton_bw_cross_attention as _xa

    return _xa


try:
    xa = _load()
    BwdVariant = xa.BwdVariant
    HAS_HSTU_CROSS_ATTN = True
    IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - report the reason, keep the op discoverable
    xa = None
    BwdVariant = None
    HAS_HSTU_CROSS_ATTN = False
    IMPORT_ERROR = e
