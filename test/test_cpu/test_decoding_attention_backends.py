# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "blackwell,has_mslk,has_flash_v3,expected",
    [
        (False, True, True, True),
        (True, True, True, False),
        (False, False, True, True),
        (False, True, False, False),
    ],
)
def test_mslk_import_only_disables_fa3_on_blackwell(
    blackwell, has_mslk, has_flash_v3, expected
):
    # Import in a child so the benchmark registrations do not leak between cases.
    program = textwrap.dedent(
        """
        import sys
        import types
        import tritonbench.utils.env_utils as env

        blackwell, has_mslk, has_flash_v3, expected = [x == "1" for x in sys.argv[1:]]
        env.IS_BLACKWELL = blackwell
        env.is_b200 = lambda: blackwell
        sys.modules["flash_attn_interface"] = (
            types.ModuleType("flash_attn_interface") if has_flash_v3 else None
        )
        sys.modules["ai_codesign"] = None
        if has_mslk:
            for name in ("mslk", "mslk.attention", "mslk.attention.cutlass_blackwell_fmha"):
                package = types.ModuleType(name)
                package.__path__ = []
                sys.modules[name] = package
            name = "mslk.attention.cutlass_blackwell_fmha.cutlass_blackwell_fmha_interface"
            sys.modules[name] = types.ModuleType(name)
        else:
            sys.modules["mslk"] = None
        import tritonbench.operators.decoding_attention.operator as op
        assert op.HAS_FLASH_V3 is expected, (op.HAS_FLASH_V3, expected)
        """
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            *[str(int(x)) for x in (blackwell, has_mslk, has_flash_v3, expected)],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
