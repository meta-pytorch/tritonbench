# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tools.mslk import install


@pytest.mark.parametrize(
    "requested,expected",
    [
        (None, "9.0;9.0a;10.0;10.0a;12.0"),
        ("10.3a", "10.3a"),
        ("9.0a;10.3a", "9.0a;10.3a"),
    ],
)
def test_source_build_honors_cuda_arch_list(monkeypatch, requested, expected):
    if requested is None:
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    else:
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", requested)
    monkeypatch.setattr(install.os.path, "exists", lambda path: True)
    monkeypatch.setattr(install, "is_hip", lambda: False)
    monkeypatch.setattr(
        install, "pip_install_requirements", lambda *args, **kwargs: None
    )
    commands = []
    monkeypatch.setattr(
        install.subprocess, "check_call", lambda cmd, **kwargs: commands.append(cmd)
    )

    install.install_build_mslk()
    assert len(commands) == 1
    assert f"-DTORCH_CUDA_ARCH_LIST={expected}" in commands[0]
