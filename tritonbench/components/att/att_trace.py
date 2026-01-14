import logging
import os
import subprocess
from pathlib import Path
from typing import List

from tritonbench.utils.env_utils import is_fbcode

target_kernel = os.getenv("TRITONBENCH_ATT_KERNEL", None)
if target_kernel:
    kernel_text = f"KERNEL={target_kernel}"
else:
    kernel_text = "range: 0:20"

# as an example, we only capture the first 20 kernels per operator
INPUT_TXT_TEMPLATE = f"""
att: TARGET_CU=1
SE_MASK=0x1
SIMD_SELECT=0xF
ISA_CAPTURE_MODE=2
{kernel_text}
"""

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _generate_input_txt(att_profile_dir: str) -> str:
    input_file_path = os.path.join(att_profile_dir, "input.txt")
    with open(input_file_path, "w") as f:
        f.write(INPUT_TXT_TEMPLATE)
    return input_file_path


def launch_att(att_profile_dir: str, benchmark_cmd: List[str]) -> str:
    run_env = os.environ.copy()
    p = Path(att_profile_dir)
    p.mkdir(parents=True, exist_ok=True)
    input_file_path = _generate_input_txt(att_profile_dir)
    if is_fbcode():
        profiler = "fbrocrof"
    else:
        profiler = "rocprofv2"
    cmd = [
        profiler,
        "-i",
        input_file_path,
        "-d",
        str(att_profile_dir.absolute()),
        "--plugin",
        "att",
        "auto",
        "--mode",
        "file,csv",
    ]
    if is_fbcode():
        from tritonbench.utils.fb.rocm_utils import ROCP_PRELOAD_LIBS

        run_env["ROCP_PRELOAD"] = ROCP_PRELOAD_LIBS
    cmd.extend(benchmark_cmd)
    logger.info("launching ATT tracer: " + " ".join(cmd))
    subprocess.check_call(cmd, env=run_env)
    return str(att_profile_dir.absolute())
