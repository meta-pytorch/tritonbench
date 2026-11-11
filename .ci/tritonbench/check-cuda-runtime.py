import ctypes
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

import torch
from packaging.requirements import Requirement
from packaging.version import Version

requirements = Path(__file__).with_name("cuda-runtime-overrides.txt")
versions = {}
for line in requirements.read_text().splitlines():
    requirement = Requirement(line)
    version = metadata.version(requirement.name)
    assert (
        version in requirement.specifier
    ), f"{requirement.name}=={version}, expected {requirement}"
    versions[requirement.name] = version

assert (
    torch.version.cuda == "13.4"
), f"Expected CUDA 13.4 Torch, got {torch.__version__}"
expected = Version(versions["nvidia-cublas"]).release[:3]
cublas = ctypes.CDLL("libcublas.so.13")
cublas.cublasGetProperty.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
cublas.cublasGetProperty.restype = ctypes.c_int
loaded = []
for property_id in range(3):
    value = ctypes.c_int()
    status = cublas.cublasGetProperty(property_id, ctypes.byref(value))
    assert status == 0, f"cublasGetProperty failed: {status}"
    loaded.append(value.value)
assert tuple(loaded) == expected, f"Loaded cuBLAS {loaded}, expected {expected}"
cublaslt = ctypes.CDLL("libcublasLt.so.13")
cublaslt.cublasLtGetVersion.restype = ctypes.c_size_t
lt_version = cublaslt.cublasLtGetVersion()
assert lt_version == expected[0] * 10000 + expected[1] * 100 + expected[2], lt_version

print(
    json.dumps(
        {
            "runtime": os.environ["TRITONBENCH_CUDA_RUNTIME"],
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "packages": versions,
            "loaded_cublas": loaded,
            "loaded_cublaslt": lt_version,
        },
        indent=2,
    )
)
