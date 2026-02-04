"""
Measure and collect compile time for operators.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from ..common import setup_output_dir, setup_tritonbench_cwd

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
    from tritonbench.metadata.query import get_benchmark_dtype

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
    dtype = get_benchmark_dtype(op)
    return {
        f"{dtype}_{op}_fwd": {"op": op, "args": command_args},
        f"{dtype}_{op}_bwd": {"op": op, "args": bwd_command_args},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="compile_time", help="Benchmark name.")
    parser.add_argument(
        "--ci", action="store_true", help="Running in GitHub Actions CI mode."
    )
    parser.add_argument(
        "--op", required=False, default=None, help="Run a single operator."
    )
    parser.add_argument(
        "--log-scuba", action="store_true", help="Upload results to Scuba."
    )
    args = parser.parse_args()
    setup_tritonbench_cwd()

    from tritonbench.utils.run_utils import run_in_task
    from tritonbench.utils.scuba_utils import decorate_benchmark_data, log_benchmark

    output_files = []
    run_timestamp, output_dir = setup_output_dir("compile_time", ci=args.ci)
    operator_benchmarks = {}
    if args.op:
        op_list = [args.op]
    else:
        op_list = TRITON_OPERATORS.keys()
    for op in op_list:
        operator_benchmarks.update(get_common_args(op, TRITON_OPERATORS[op]))
    for op_bench in operator_benchmarks:
        op_args = operator_benchmarks[op_bench]["args"]
        output_file = output_dir.joinpath(f"{op_bench}.json")
        op_args.extend(["--output-json", str(output_file.absolute())])
        run_in_task(op_args=op_args, benchmark_name=op_bench)
        # write pass or fail to result json
        # todo: check every input shape has passed
        output_file_name = Path(output_file).stem
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logger.warning(f"[compile_time] Failed to run {output_file_name}.")
            with open(output_file, "w") as f:
                json.dump({f"tritonbench_{output_file_name}-pass": 0}, f)
        else:
            with open(output_file, "r") as f:
                obj = json.load(f)
            obj[f"tritonbench_{output_file_name}-pass"] = 1
            with open(output_file, "w") as f:
                json.dump(obj, f, indent=4)
        output_files.append(output_file)
    # Reduce all operator CSV outputs to a single output json
    benchmark_data = [json.load(open(f, "r")) for f in output_files]
    aggregated_obj = decorate_benchmark_data(
        args.name, run_timestamp, args.ci, benchmark_data
    )
    result_json_file = os.path.join(output_dir, "result.json")
    with open(result_json_file, "w") as fp:
        json.dump(aggregated_obj, fp, indent=4)
    logger.info(f"[compile_time] logging result json file to {result_json_file}.")
    if args.log_scuba:
        log_benchmark(aggregated_obj)
        logger.info(f"[compile_time] logging results to scuba.")


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
