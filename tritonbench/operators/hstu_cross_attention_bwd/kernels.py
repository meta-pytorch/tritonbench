# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Import shim for the HSTU cross-attention backward kernel.

The kernel lives in the Triton tree at
``third_party/tlx/tutorials/hstu_cross_attn/triton_bw_cross_attention.py`` and is
exposed through the installed Triton package as
``triton.language.extra.tlx.tutorials.hstu_cross_attn`` (a symlink to that tree).
It is written as a standalone script rather than a package: it uses
``sys.path``-relative imports (``import stubs``, ``triton_attention_utils``) and
gates ``_HSTU_COMPUTE_FOLD`` on an env var read at *import* time. This shim sets
that env and puts the kernel directory on ``sys.path`` before importing, so the
operator can just ``from .kernels import xa, BwdVariant``.
"""

import os
import sys

# The 2-KV data-partition variants (tlx_2kv / autows_2kv) require compute-fold,
# which the kernel reads as a module-level tl.constexpr at import time -- so it
# must be set BEFORE the import below. TRITON_ALLOW_NON_CONSTEXPR_GLOBALS lets the
# module use its module-level (non-constexpr) global config objects.
os.environ.setdefault("HSTU_COMPUTE_FOLD", "1")
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")


def _load():
    import triton

    kernel_dir = os.path.join(
        os.path.dirname(triton.__file__),
        "language",
        "extra",
        "tlx",
        "tutorials",
        "hstu_cross_attn",
    )
    if not os.path.isdir(kernel_dir):
        raise ImportError(
            f"HSTU cross-attention kernels not found at {kernel_dir}; "
            "this operator requires a Triton build that ships the tlx tutorials."
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
