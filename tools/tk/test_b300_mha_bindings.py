import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_PATH = Path(__file__).resolve().parents[2]
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from tools.tk.install import TK_TOOLS_PATH, _ensure_pybind11, _ext_suffix, _get_env

MODULE_SPECS = (
    {
        "name": "bf16_b300_mha_causal",
        "makefile": TK_TOOLS_PATH / "bf16_b300_mha_causal.Makefile",
        "exports": ("forward", "forward_persistent"),
    },
    {
        "name": "bf16_b300_mha_noncausal",
        "makefile": TK_TOOLS_PATH / "bf16_b300_mha_noncausal.Makefile",
        "exports": ("forward",),
    },
)


def _supports_b300_compile() -> bool:
    try:
        archs = subprocess.check_output(["nvcc", "--list-gpu-arch"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return "compute_103a" in archs.split()


def _build_extension(makefile: Path, output_dir: Path, gpu: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"_C{_ext_suffix()}"
    if output_path.exists():
        output_path.unlink()

    subprocess.check_call(
        [
            "make",
            "-f",
            str(makefile),
            f"GPU={gpu}",
            f"OUT={output_path}",
            "CONFIG=pytorch",
        ],
        cwd=TK_TOOLS_PATH,
        env=_get_env(),
    )
    return output_path


def _assert_exports(module_dir: Path, expected_exports: tuple[str, ...]) -> None:
    sys.path.insert(0, str(module_dir))
    try:
        module = importlib.import_module("_C")
        missing = [name for name in expected_exports if not hasattr(module, name)]
        if missing:
            raise AssertionError(f"missing exports: {missing}")
    finally:
        sys.path.pop(0)
        sys.modules.pop("_C", None)


def _is_b300() -> bool:
    return torch.cuda.is_available() and "B300" in torch.cuda.get_device_name(0)


def _make_inputs():
    q = torch.randn((1, 128, 1, 192), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 128, 1, 192), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, 128, 1, 128), device="cuda", dtype=torch.bfloat16)
    o = torch.empty((1, 128, 1, 128), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((1, 1, 1, 128), device="cuda", dtype=torch.float32)
    return q, k, v, o, lse


def _run_b300_smoke(module_dir: Path, expected_exports: tuple[str, ...]) -> None:
    sys.path.insert(0, str(module_dir))
    try:
        module = importlib.import_module("_C")
        q, k, v, o, lse = _make_inputs()
        module.forward(q, k, v, o, lse)
        torch.cuda.synchronize()
        if "forward_persistent" in expected_exports:
            module.forward_persistent(q, k, v, o, lse)
            torch.cuda.synchronize()
    finally:
        sys.path.pop(0)
        sys.modules.pop("_C", None)


def main() -> None:
    _ensure_pybind11()
    with tempfile.TemporaryDirectory(prefix="tk_b300_bindings_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        if _supports_b300_compile():
            print("testing native B300 build path")
            for spec in MODULE_SPECS:
                module_dir = tmpdir_path / f"{spec['name']}_b300"
                output_path = _build_extension(spec["makefile"], module_dir, gpu="B300")
                _assert_exports(module_dir, spec["exports"])
                print(f"verified exports for {spec['name']}: {', '.join(spec['exports'])}")
                if _is_b300():
                    _run_b300_smoke(module_dir, spec["exports"])
                    print(
                        f"ran runtime smoke for {spec['name']} using {output_path.name}"
                    )
                else:
                    print(
                        f"skipped runtime smoke for {spec['name']}: current GPU is not B300"
                    )
        else:
            print("skipped native B300 build test: nvcc does not support compute_103a")


if __name__ == "__main__":
    main()
