"""
Measure and collect compile time for operators.
"""

import argparse
import json
import logging
import os
import sys

from os.path import abspath, exists
from typing import Dict, List


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_tritonbench_cwd():
    original_dir = abspath(os.getcwd())

    for tritonbench_dir in (
        ".",
        "../../../tritonbench",
    ):
        if exists(tritonbench_dir):
            break

    if exists(tritonbench_dir):
        tritonbench_dir = abspath(tritonbench_dir)
        os.chdir(tritonbench_dir)
        sys.path.append(tritonbench_dir)
    return original_dir


# A list of operators and their Triton backends
TRITON_OPERATORS = {
    "addmm": ["triton_addmm"],
    "bf16xint16_gemm": ["bf16xbf16"],
    "cross_entropy": ["liger_cross_entropy_loss"],
    "embedding": ["liger_embedding"],
    "flash_attention": ["triton_tutorial_flash_v2"],
    "fp8_attention": ["triton_flash_v2_tma"],
    "fp8_fused_quant_gemm_rowwise": ["rms_norm_fused"],
    "fp8_gemm": ["triton_tma_persistent_fp8_gemm"],
    "fp8_gemm_blockwise": ["_triton"],
    "fp8_gemm_rowwise": ["_triton"],
    "fused_linear_cross_entropy": ["liger_lm_head_ce"],
    "fused_linear_jsd": ["liger_lm_head_jsd"],
    "geglu": ["liger_geglu"],
    "gemm": ["triton_tutorial_matmul"],
    "grouped_gemm": ["triton"],
    "int4_gemm": ["triton"],
    "jsd": ["liger_jsd"],
    "kl_div": ["liger_kl_div"],
    "layer_norm": ["liger_layer_norm"],
    "low_mem_dropout": ["triton_dropout"],
    "ragged_attention": ["hstu_triton_ragged_attention"],
    "rms_norm": ["liger_rms"],
    "rope": ["liger_rotary_pos_emb"],
    "softmax": ["triton_softmax"],
    "swiglu": ["liger_swiglu"],
    "template_attention": ["test_no_exp2"],
    "welford": ["test_welford"],
}


def get_common_args(op: str, backends: List[str]) -> Dict[str, List[str]]:
    command_args = [
        "--op",
        op,
        "--only",
        ",".join(backends),
        "--num-inputs",
        "1",
        "--metrics",
        "compile_time",
    ]
    bwd_command_args = command_args.copy()
    bwd_command_args.append("--bwd")
    return {"fwd": command_args, "bwd": bwd_command_args}


def reduce(run_timestamp, output_dir, output_files, args):
    """aggregate all op benchmark csvs into json file"""
    from tritonbench.utils.path_utils import REPO_PATH
    from tritonbench.utils.run_utils import get_github_env, get_run_env

    repo_locs = {
        "tritonbench": REPO_PATH,
    }
    if args.ci and "TRITONBENCH_TRITON_REPO_PATH" in os.environ:
        repo_locs["triton"] = os.environ.get("TRITONBENCH_TRITON_REPO_PATH", None)
        repo_locs["pytorch"] = os.environ.get("TRITONBENCH_PYTORCH_REPO_PATH", None)
    aggregated_obj = {
        "name": "compile_time",
        "env": get_run_env(run_timestamp, repo_locs),
        "metrics": {},
    }
    # Collecting GitHub environment variables when running in CI environment
    if args.ci:
        aggregated_obj["github"] = get_github_env()

    for result_json_file in output_files:
        if not exists(result_json_file):
            logger.warning(
                f"[compile_time] result json file {result_json_file} does not exist."
            )
            continue
        with open(
            result_json_file,
            "r",
        ) as fp:
            result_obj = json.load(fp)
            aggregated_obj["metrics"].update(result_obj)
    result_json_path = os.path.join(output_dir, "result.json")
    with open(result_json_path, "w") as fp:
        json.dump(aggregated_obj, fp, indent=4)
    return result_json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ci", action="store_true", help="Running in GitHub Actions CI mode."
    )
    parser.add_argument(
        "--op", required=False, default=None, help="Run a single operator."
    )
    args = parser.parse_args()
    setup_tritonbench_cwd()

    from tritonbench.utils.run_utils import run_in_task, setup_output_dir

    output_files = []
    run_timestamp, output_dir = setup_output_dir("compile_time")
    op_args_list = {}
    if args.op:
        op_list = [args.op]
    else:
        op_list = TRITON_OPERATORS.keys()
    for op in op_list:
        op_args_list[op] = get_common_args(op, TRITON_OPERATORS[op])
    for op in op_list:
        for mode in op_args_list[op]:
            op_args = op_args_list[op][mode]
            output_file = output_dir.joinpath(f"{op}_{mode}.json")
            op_args.extend(["--output-json", str(output_file.absolute())])
            run_in_task(op=op, op_args=op_args)
            output_files.append(output_file)
    # Reduce all operator CSV outputs to a single output json
    result_json_file = reduce(run_timestamp, output_dir, output_files, args)
    logger.info(f"[compile_time] logging result json file to {result_json_file}.")
