import ctypes
import os


def preload_cuda_driver():
    candidates = []
    for path in (
        "/usr/lib64/libcuda.so.1",
        "/lib64/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    ):
        if os.path.exists(path):
            candidates.append(path)

    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        stub = os.path.join(cuda_home, "lib64", "stubs", "libcuda.so")
        if os.path.exists(stub):
            candidates.append(stub)

    for path in candidates:
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return path
        except OSError:
            continue

    return None
