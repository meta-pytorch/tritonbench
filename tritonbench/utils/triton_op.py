import argparse
import copy
import csv
import functools
import hashlib
import json
import logging
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from collections import defaultdict, OrderedDict
from dataclasses import asdict, dataclass, fields
from enum import Enum
from itertools import product
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import psutil
import tabulate
import torch
import triton

from tritonbench.components.do_bench import (
    do_bench_cuda_time,
    do_bench_wrapper,
    Latency,
)
from tritonbench.components.export import export_data

from tritonbench.utils.env_utils import (
    apply_precision,
    is_fbcode,
    is_hip,
    set_env,
    set_random_seed,
)
from tritonbench.utils.input import input_cast
from tritonbench.utils.path_utils import add_cmd_parameter, remove_cmd_parameter

if is_hip():
    from tritonbench.components.att import launch_att
else:
    from tritonbench.components.ncu import ncu_analyzer, nsys_analyzer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class BenchmarkOperatorBackend:
    # backend name
    name: str
    # backend label
    label: str
    # baseline
    baseline: bool = False
    # enabled
    enabled: bool = True
    # fwd_only
    # if an operator supports backward, but one of the kernels do not
    # set fwd_only = True
    fwd_only: bool = False
    # need to be tested in ci
    # ci = False implies enabled = False
    ci: bool = True


DEFAULT_WARMUP = 25
DEFAULT_RUN_ITERS = 100
DEFAULT_QUANTILES = [0.5, 0.1, 0.9]
REGISTERED_BENCHMARKS: Dict[str, OrderedDict[str, BenchmarkOperatorBackend]] = {}
REGISTERED_METRICS: defaultdict[str, List[str]] = defaultdict(list)
OVERRIDDEN_METRICS: defaultdict[str, List[str]] = defaultdict(list)
REGISTERED_X_VALS: Dict[str, str] = {}
BASELINE_BENCHMARKS: Dict[str, str] = {}
BASELINE_SKIP_METRICS = {
    "speedup",
    "accuracy",
    "mem_footprint_compression_ratio",
    "nsys_gpu_speedup",
}
X_ONLY_METRICS = set(["hw_roofline"])
PRECISION_DTYPE_MAPPING = {
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_RANGE_NAME = "tritonbench_range"


class Mode(Enum):
    FWD = "fwd"
    BWD = "bwd"
    FWD_BWD = "fwd_bwd"
    FWD_NO_GRAD = "fwd_no_grad"


class TimerContext:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.elapsed_ms = None

    def __enter__(self):
        if self.enabled:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        if self.enabled:
            end_time = time.perf_counter()
            self.elapsed_ms = (end_time - self._start_time) * 1e3


def do_bench_walltime(fn, warmup=25, rep=100):
    fn()
    torch.cuda.synchronize()

    with TimerContext() as timer:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    estimate_ms = timer.elapsed_ms / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    wall_time_ms = (end_time - start_time) * 1e3 / n_repeat
    return wall_time_ms


def gemm_shapes(prefill: bool = False):
    """Gets an extensive list of GEMM shapes for benchmarking"""
    if not is_fbcode():
        return
    from .fb.fp8_gemm_rowwise_shapes import read_shapes_for_fp8_gemm_rowwise

    return read_shapes_for_fp8_gemm_rowwise(prefill)


def llama_shapes():
    # batch sizes * seq lengths
    BS = [2**i for i in range(0, 17)]
    # attn: wqkv, wo; ffn: w13, w2
    KN = [
        (4096, 12288),
        (4096, 4096),
        (4096, 22016),
        (11008, 4096),
        (8192, 1280),
        (1024, 8192),
        (8192, 7168),
        (3584, 8192),
        (16384, 2304),
        (2048, 16384),
        (16384, 13312),
        (6656, 16384),
    ]
    return [(bs, n, k, None) for bs, (k, n) in product(BS, KN)]


def _split_params_by_comma(params: Optional[str]) -> List[str]:
    if params == None:
        return []
    return [x.strip() for x in params.split(",")] if "," in params else [params]


def _find_op_name_from_module_path(module_path: str) -> str:
    PATH_PREFIX = "tritonbench.operators."
    # We have a separate operator loader for aten operator benchmark.
    PATH_PREFIX_LOADER = "tritonbench.operator_loader.loaders."
    assert (
        PATH_PREFIX in module_path or PATH_PREFIX_LOADER in module_path
    ), f"We rely on module path prefix to identify operator name. Expected {PATH_PREFIX}<operator_name>, get {module_path}."
    if PATH_PREFIX_LOADER in module_path:
        suffix = module_path.partition(PATH_PREFIX_LOADER)[2]
    else:
        suffix = module_path.partition(PATH_PREFIX)[2]
    if suffix.startswith("fb."):
        return suffix.split(".")[1]
    return suffix.split(".")[0]


@dataclass
class BenchmarkOperatorMetrics:
    # latency
    latency: Optional[Latency] = None
    # tflops
    tflops: Optional[float] = None
    # speedup over baseline
    speedup: Optional[float] = None
    # accuracy over baseline
    accuracy: Optional[bool] = None
    # wall time
    walltime: Optional[float] = None
    # compile time
    compile_time: Optional[float] = None
    # stage breakdown of compile times
    compile_time_by_stage: Optional[Dict[str, float]] = None
    # compile time with kineto trace
    compile_trace: Optional[str] = None
    # att trace directory
    att_trace: Optional[str] = None
    # ncu replay file
    ncu_rep: Optional[str] = None
    # ncu replay file with TTGIR line numbers
    ncu_rep_ir: Optional[str] = None
    # nsys replay file
    nsys_rep: Optional[str] = None
    # kineto trace file
    kineto_trace: Optional[str] = None
    # cpu peak memory
    cpu_peak_mem: Optional[float] = None
    # gpu peak memory
    gpu_peak_mem: Optional[float] = None
    # error message
    error_msg: Optional[str] = None
    # hw roofline
    hw_roofline: Optional[float] = None
    # best config
    best_config: Optional[Dict[str, Any]] = None
    # all configs
    all_configs: Optional[str] = None
    # extra metrics
    extra_metrics: Optional[Dict[str, float]] = None
    # mem footprint
    mem_footprint_compression_ratio: Optional[float] = None
    # speedup for the summary of kernel GPU time only
    nsys_gpu_speedup: Optional[float] = None
    # hashed source code for the kernel
    kernel_source_hash: Optional[str] = None
    # cuda time
    cuda_time: Optional[float] = None


BUILTIN_METRICS = {x.name for x in fields(BenchmarkOperatorMetrics)} - {"extra_metrics"}


@dataclass
class BenchmarkOperatorResult:
    # Print the result in a table format
    benchmark_name: str
    op_name: str
    op_mode: str
    metrics: List[str]
    simple_mode: bool
    # Tuple: (x_val, Dict[impl_name, BenchmarkOperatorMetrics])
    result: List[Tuple[Any, Dict[str, BenchmarkOperatorMetrics]]]
    _result_dict: Optional[Dict[Number, Dict[str, BenchmarkOperatorMetrics]]] = None

    def _table(self):
        table = []
        # generate headers
        headers = [REGISTERED_X_VALS[self.op_name]]
        if len(self.result) == 0:
            return headers, table
        y_val = self.result[0][1]
        backends = list(y_val.keys())
        # move the baseline benchmark to the front of the list if exists
        if (
            self.op_name in BASELINE_BENCHMARKS
            and BASELINE_BENCHMARKS[self.op_name] in backends
        ):
            backends.insert(
                0, backends.pop(backends.index(BASELINE_BENCHMARKS[self.op_name]))
            )
        key_metrics = {}
        # Add header for x_only_metrics
        x_only_metrics = sorted(
            [metric for metric in self.metrics if metric in X_ONLY_METRICS]
        )
        headers.extend(x_only_metrics)
        for backend in backends:
            label = REGISTERED_BENCHMARKS[self.op_name][backend].label

            def select_metric(backend, m):
                if m in x_only_metrics:
                    return False
                if (
                    m in BASELINE_SKIP_METRICS
                    and backend == BASELINE_BENCHMARKS[self.op_name]
                ):
                    return False
                if m == "all_configs":
                    return False
                return True

            key_metrics[backend] = [
                metric for metric in self.metrics if select_metric(backend, metric)
            ]
            for metric in key_metrics[backend]:
                # add extra metrics
                headers.append(f"{label}-{metric}")
        # generate rows
        hashes = {}
        if "kernel_source_hash" in self.metrics:
            self.result.append(tuple(["hashes", {}]))
        avg_row = []
        for x_val, y_val in self.result:
            col_num = 0
            row = []
            row.append(x_val)
            # Append x_only metrics
            for x_only_metric in x_only_metrics:
                if x_val == "hashes" and len(hashes) > 0:
                    continue
                next_val = None
                # retrieve x_only metrics from the first backend metrics
                x_only_metric_dict = asdict(y_val[backends[0]])
                if (
                    "extra_metrics" in x_only_metric_dict
                    and x_only_metric in x_only_metric_dict["extra_metrics"]
                ):
                    next_val = x_only_metric_dict["extra_metrics"][x_only_metric]

                else:
                    next_val = x_only_metric_dict[x_only_metric]
                row.append(next_val)
                if len(avg_row) <= col_num:
                    avg_row.append(next_val if isinstance(next_val, Number) else None)
                else:
                    if avg_row[col_num] is None:
                        avg_row[col_num] = next_val
                    elif isinstance(next_val, Number):
                        avg_row[col_num] = avg_row[col_num] + next_val
                col_num += 1
            for backend in backends:
                if x_val == "hashes" and len(hashes) > 0:
                    row.append(hashes[backend])
                    continue
                metrics_dict = asdict(y_val[backend])
                if "kernel_source_hash" in metrics_dict:
                    hashes[backend] = metrics_dict.pop("kernel_source_hash")
                if metrics_dict["error_msg"]:
                    # Add error message to the display row
                    row.append(metrics_dict["error_msg"])
                    row.extend([None] * (len(key_metrics[backend]) - 1))

                    # Skip this backend's metrics in the average row to maintain alignment
                    num_metrics_to_skip = len(key_metrics[backend])
                    for _ in range(num_metrics_to_skip):
                        if len(avg_row) <= col_num:
                            avg_row.append(None)
                        col_num += 1
                    continue
                for metric in key_metrics[backend]:
                    _metrics_dict = (
                        metrics_dict["extra_metrics"]
                        if metric in metrics_dict["extra_metrics"]
                        else metrics_dict
                    )
                    metric_val = _metrics_dict.get(metric, None)
                    row.append(metric_val)
                    if len(avg_row) <= col_num:
                        avg_row.append(
                            metric_val if isinstance(metric_val, Number) else None
                        )
                    else:
                        if avg_row[col_num] is None:
                            avg_row[col_num] = metric_val
                        elif isinstance(metric_val, Number):
                            avg_row[col_num] = avg_row[col_num] + metric_val
                    col_num += 1
            table.append(row)

        if self.simple_mode:
            # do not print header if in simple output mode
            avg_row = [
                str(x / len(self.result)) if isinstance(x, Number) else None
                for x in avg_row
            ]
            avg_row = [",".join(avg_row)]
        else:
            avg_row = ["average"] + [
                x / len(self.result) if isinstance(x, Number) else None for x in avg_row
            ]
        table.append(avg_row)
        return headers, table

    def _post_process_table(self, table, style="plain"):
        """
        The "plain" style will use p50 for all List or Latency metrics.
        The "with_variance" style will use "with_variance" str for Latency.
        """

        def _inner(table_cell):
            if isinstance(table_cell, list):
                # Check if all elements are numbers before calculating median
                if all(isinstance(x, Number) for x in table_cell):
                    return statistics.median(table_cell)
                else:
                    # For non-numeric lists, convert to string representation
                    table_cell_str = str(table_cell)
                    if ";" in table_cell_str:
                        logger.warning(
                            f"Metric value '{table_cell_str}' contains semicolon which may cause CSV parsing issues"
                        )
                    return table_cell_str
            elif isinstance(table_cell, Latency):
                # print variance to latency metric
                return (
                    table_cell.to_str(mode=style)
                    if style == "with_variance"
                    else str(table_cell)
                )
            elif isinstance(table_cell, bool):
                return 1.0 if table_cell else 0.0
            elif isinstance(table_cell, str):
                if ";" in table_cell:
                    logger.warning(
                        f"Metric value '{table_cell}' contains semicolon which may cause CSV parsing issues"
                    )
                return table_cell
            else:
                return table_cell

        return [[_inner(cell) for cell in row] for row in table]

    def write_csv_to_file(self, fileobj):
        headers, table = self._table()
        table = self._post_process_table(table)
        writer = csv.writer(fileobj, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(table)

    def write_csv(self, dir_path):
        # This is just a way to create a unique filename. It's not actually a
        # temporary file (since delete=False).
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=os.path.join(dir_path, f"op_{self.op_name}_"),
            suffix=".csv",
            newline="",
            delete=False,
        ) as fileobj:
            self.write_csv_to_file(fileobj)
            return fileobj.name

    def write_json_to_file(self, fileobj):
        json.dump(self.userbenchmark_dict, fileobj, indent=4)

    @property
    def x_vals(self):
        return sorted(self._get_result_dict().keys())

    @property
    def userbenchmark_dict(self) -> Dict[str, Any]:
        # Userbenchmark Metric key format:
        # tritonbench_{op_name}_{op_mode}[{x_val}-{provider}-{metric}]
        userbenchmark_metrics_dict = {}
        headers, table = self._table()
        table = self._post_process_table(table)
        agg_data = {}
        benchmark_name = (
            self.benchmark_name
            if self.benchmark_name
            else f"{self.op_name}_{self.op_mode}"
        )
        for row in table:
            x_val = row[0]

            for ind, v in enumerate(row[1:]):
                header = headers[ind + 1]
                provider, _, metrics_name = header.partition("-")
                metrics_dict = {metrics_name: v}
                if v and isinstance(v, dict):
                    metrics_dict = {
                        f"{metrics_name}-{k}": value for k, value in v.items()
                    }
                for metrics, value in metrics_dict.items():
                    metric_name = (
                        f"tritonbench_{benchmark_name}[x_{x_val}-{provider}]_{metrics}"
                    )
                    userbenchmark_metrics_dict[metric_name] = value
                    agg_metric_name = (
                        f"tritonbench_{benchmark_name}[{provider}]-{metrics}-avg"
                    )
                    if value is None:
                        continue
                    if isinstance(value, (int, float)):
                        agg_data[agg_metric_name] = agg_data.get(
                            agg_metric_name, []
                        ) + [value]
        final_agg_data = {k: sum(v) / len(v) for k, v in agg_data.items()}
        userbenchmark_metrics_dict.update(final_agg_data)

        return userbenchmark_metrics_dict

    def get_y_vals(self, x_val, provider, metric_name: str):
        if provider in X_ONLY_METRICS:
            maybe_baseline = list(REGISTERED_BENCHMARKS[self.op_name].keys())[0]
            metrics_dict = asdict(self._get_result_dict()[x_val][maybe_baseline])
            metric_name = provider
        else:
            y_vals = self._get_result_dict()[x_val][provider]
            metrics_dict = asdict(y_vals)
        if metric_name in metrics_dict:
            return metrics_dict[metric_name]
        assert (
            metric_name in metrics_dict["extra_metrics"]
        ), f"Metric {metric_name} could not be found."
        return metrics_dict["extra_metrics"][metric_name]

    def _get_result_dict(self):
        if not self._result_dict:
            self._result_dict = {}
            for x_val, y_val in self.result:
                self._result_dict[x_val] = y_val
        return self._result_dict

    def __str__(self):
        headers, table = self._table()
        table = self._post_process_table(table, style="with_variance")
        table = tabulate.tabulate(table, headers=headers, stralign="right")
        return table


def find_enabled_benchmarks(mode, benchmark_backends, skip_benchmarks):
    """Condition: enabled, not skipped"""

    def runnable(m, backend):
        return (not (m == Mode.BWD or m == Mode.FWD_BWD)) or not backend.fwd_only

    if skip_benchmarks:
        # even if we are skipping benchmarks, we have to check if the backend is enabled
        # e.g., we can't run CUTLASS on AMD even if it isn't explicitly skipped
        benchmarks = [
            bm
            for bm in benchmark_backends.keys()
            if bm not in skip_benchmarks
            and runnable(mode, benchmark_backends[bm])
            and benchmark_backends[bm].enabled
        ]
    else:
        benchmarks = [
            bm
            for bm in benchmark_backends.keys()
            if benchmark_backends[bm].enabled and runnable(mode, benchmark_backends[bm])
        ]
    return benchmarks


def register_x_val(label: str = "x_val"):
    def decorator(function):
        operator_name = _find_op_name_from_module_path(function.__module__)
        REGISTERED_X_VALS[operator_name] = label

        def _inner(self, *args, **kwargs):
            return function(self, *args, **kwargs)

        return _inner

    return decorator


def register_benchmark(
    operator_name: Optional[str] = None,
    func_name: Optional[str] = None,
    baseline: bool = False,
    enabled: bool = True,
    fwd_only: bool = False,
    label: Optional[str] = None,
):
    def decorator(function):
        op_name = (
            _find_op_name_from_module_path(function.__module__)
            if not operator_name
            else operator_name
        )
        fn_name = function.__name__ if not func_name else func_name
        backend_config = BenchmarkOperatorBackend(
            name=fn_name,
            label=label if label else fn_name,
            baseline=baseline,
            enabled=enabled,
            fwd_only=fwd_only,
        )
        if op_name not in REGISTERED_BENCHMARKS:
            REGISTERED_BENCHMARKS[op_name] = OrderedDict()
        REGISTERED_BENCHMARKS[op_name][fn_name] = backend_config
        if backend_config.baseline:
            BASELINE_BENCHMARKS[op_name] = fn_name

        def _inner(self, *args, **kwargs):
            return function(self, *args, **kwargs)

        return _inner

    return decorator


def register_metric(
    # Metrics that only apply to non-baseline impls
    # E.g., accuracy, speedup
    skip_baseline: bool = False,
    # Metrics that are the same across all impls
    # E.g., x_shape, hw_roofline
    x_only: bool = False,
):
    def decorator(func):
        metric_name = func.__name__
        operator_name = _find_op_name_from_module_path(func.__module__)
        if metric_name not in BUILTIN_METRICS:
            REGISTERED_METRICS[operator_name].append(func.__name__)
        else:
            OVERRIDDEN_METRICS[operator_name].append(metric_name)
        if skip_baseline:
            BASELINE_SKIP_METRICS.add(func.__name__)
        if x_only:
            X_ONLY_METRICS.add(func.__name__)

        def _inner(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        return _inner

    return decorator


class PostInitProcessor(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj


def _translate_mode(tb_args):
    def _has_and_true(attr):
        if hasattr(tb_args, attr) and getattr(tb_args, attr):
            return True
        return False

    if _has_and_true("fwd"):
        tb_args.mode = "fwd"
    if _has_and_true("bwd"):
        tb_args.mode = "bwd"
    if _has_and_true("fwd_bwd"):
        tb_args.mode = "fwd_bwd"
    if _has_and_true("fwd_no_grad"):
        tb_args.mode = "fwd_no_grad"


class BenchmarkOperator(metaclass=PostInitProcessor):
    mode: Mode = Mode.FWD
    test: str = "eval"
    device: str = "cuda"
    # By default, do not touch the input data dtype
    DEFAULT_PRECISION = "bypass"
    # By default, only collect latency metrics
    # Each operator can override to define their own default metrics
    DEFAULT_METRICS = ["latency"]
    required_metrics: List[str]
    _cur_input_id: Optional[int] = None
    _input_iter: Optional[Generator] = None
    extra_args: List[str] = []
    example_inputs: Any = None
    use_cuda_graphs: bool = False
    is_compute_bound = True
    # reset dynamo to avoid errors like https://github.com/meta-pytorch/tritonbench/issues/90
    reset_dynamo = False

    """
    A base class for adding operators to torch benchmark.
    """

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        set_env()
        set_random_seed()
        self.name = _find_op_name_from_module_path(self.__class__.__module__)
        self._raw_extra_args = copy.deepcopy(extra_args)
        self.tb_args = tb_args
        self.add_production_shapes = (
            self.tb_args.production_shapes if is_fbcode() else False
        )
        self.use_cuda_graphs = (
            self.tb_args.cudagraph if self.tb_args.cudagraph else self.use_cuda_graphs
        )
        _translate_mode(self.tb_args)
        if self.tb_args.mode == "fwd":
            self.mode = Mode.FWD
        elif self.tb_args.mode == "fwd_bwd":
            self.mode = Mode.FWD_BWD
        elif self.tb_args.mode == "fwd_no_grad":
            self.mode = Mode.FWD_NO_GRAD
        else:
            assert (
                self.tb_args.mode == "bwd"
            ), "We only accept test modes: fwd, bwd, fwd_bwd, or fwd_no_grad."
            self.mode = Mode.BWD
        self.requires_grad = not (self.mode == Mode.FWD_NO_GRAD)
        self.device = tb_args.device
        self.required_metrics = (
            list(dict.fromkeys(tb_args.metrics.split(",")))
            if tb_args.metrics
            else self.DEFAULT_METRICS
        )
        if "compile_time" in self.required_metrics and is_fbcode():
            self.required_metrics.append("compile_time_by_stage")
        self.extra_args = extra_args
        if self.name not in REGISTERED_X_VALS:
            REGISTERED_X_VALS[self.name] = "x_val"
        # We rely on each operator to correctly respect the input data dtype
        if self.tb_args.precision == "bypass":
            self.tb_args.precision = self.DEFAULT_PRECISION
        self.dtype = PRECISION_DTYPE_MAPPING.get(self.tb_args.precision, None)
        if self.tb_args.baseline:
            BASELINE_BENCHMARKS[self.name] = self.tb_args.baseline
        self._only = _split_params_by_comma(self.tb_args.only)
        self._skip = _split_params_by_comma(self.tb_args.skip)
        self._only_match_mode = self.tb_args.only_match_mode
        self._input_id = self.tb_args.input_id
        self._num_inputs = self.tb_args.num_inputs
        self.prod_shapes = self.tb_args.prod_shapes

    # Run the post initialization
    def __post__init__(self):
        if self.tb_args.input_loader:
            if is_fbcode() and not hasattr(self, "aten_op_name"):
                from tritonbench.data.fb.input_loader import get_input_loader

                self.get_input_iter = get_input_loader(
                    self, self.name, self.tb_args.input_loader
                )
            else:
                from tritonbench.data import get_input_loader

                self._get_input_iter = get_input_loader(
                    self, self.name, self.tb_args.input_loader
                )
        self._available_num_inputs = self.count_example_inputs()
        if self._num_inputs is None:
            self._num_inputs = self._available_num_inputs - self._input_id
        if self._num_inputs > self._available_num_inputs:
            self._num_inputs = self._available_num_inputs

    def _get_bm_func(self, bm_func_name: str):
        fwd_fn_lambda = getattr(self, bm_func_name, None)
        assert callable(fwd_fn_lambda), (
            f"Could not find benchmark {bm_func_name} registered in {self.name}. "
            f"Available benchmarks: {REGISTERED_BENCHMARKS[self.name].keys()}. "
        )
        with TimerContext(enabled=logger.level <= logging.INFO) as timer:
            if isinstance(self.example_inputs, dict):
                fwd_fn = fwd_fn_lambda(**self.example_inputs)
            else:
                fwd_fn = fwd_fn_lambda(*self.example_inputs)
        logger.info(
            "Took %.02fms to get benchmark function for %s",
            timer.elapsed_ms,
            bm_func_name,
        )

        backend = REGISTERED_BENCHMARKS[self.name][bm_func_name]
        if self.mode == Mode.FWD:
            setattr(fwd_fn, "_name", bm_func_name)
            return fwd_fn
        elif self.mode == Mode.BWD:
            assert (
                not backend.fwd_only
            ), f"Backend {bm_func_name} does not support backward pass."
            bwd_fn = self.get_bwd_fn(fwd_fn)
            setattr(bwd_fn, "_name", bm_func_name)
            return bwd_fn
        elif self.mode == Mode.FWD_BWD:
            assert (
                not backend.fwd_only
            ), f"Backend {bm_func_name} does not support backward pass."
            bwd_fn = self.get_bwd_fn(fwd_fn)
            fwd_bwd_fn = lambda: (fwd_fn(), bwd_fn())
            setattr(fwd_bwd_fn, "_name", bm_func_name)
            return fwd_bwd_fn
        elif self.mode == Mode.FWD_NO_GRAD:

            def fwd_no_grad_fn():
                with torch.no_grad():
                    fwd_fn()

            setattr(fwd_no_grad_fn, "_name", bm_func_name)
            return fwd_no_grad_fn

    def run(
        self, warmup=DEFAULT_WARMUP, rep=DEFAULT_RUN_ITERS, quantiles=DEFAULT_QUANTILES
    ) -> None:
        """Benchmarking the operator and returning its metrics."""
        metrics = []
        try:
            if "proton" in self.required_metrics:
                import triton.profiler as proton

                self._proton_session_id = proton.start()
                proton.enter_scope(f"tritonbench_run_op_{self.name}")
                proton.deactivate(self._proton_session_id)
            input_id_range = range(self._input_id, self._input_id + self._num_inputs)
            if tqdm is not None:
                input_id_range = tqdm(input_id_range)
            if self._input_id:
                for _dryrun_input_id in range(self._input_id):
                    self.example_inputs = self.get_example_inputs()
            for input_id in input_id_range:
                self._cur_input_id = input_id
                self.example_inputs = self.get_example_inputs()
                if self.reset_dynamo:
                    torch._dynamo.reset()
                x_val = self.get_x_val(self.example_inputs)
                if "proton" in self.required_metrics:
                    proton.activate(self._proton_session_id)
                    proton.enter_scope(f"x_val_{x_val}")
                    proton.deactivate(self._proton_session_id)
                if self.example_inputs is None:
                    logger.warning(
                        f"The input generator get_input_iter() has depleted at id {input_id}. Available number of "
                        f"inputs: {self._available_num_inputs}.",
                        stacklevel=1,
                    )
                    break
                # Move inputs to the device
                self.example_inputs = input_cast(
                    lambda x: isinstance(x, torch.Tensor),
                    lambda x: x.to(self.device),
                    self.example_inputs,
                )
                # Handle the input data types with best effort
                apply_precision(self, self.tb_args.precision)
                self.baseline_fn = None
                self.baseline_metrics = None
                self._op_flops = {}
                if self._only:
                    if self._only_match_mode == "prefix-with-baseline":
                        # Find all benchmarks that match any of the prefixes
                        all_benchmarks = find_enabled_benchmarks(
                            self.mode, REGISTERED_BENCHMARKS[self.name], []
                        )
                        benchmarks = []
                        for bm in all_benchmarks:
                            for prefix in self._only:
                                if bm.startswith(prefix):
                                    benchmarks.append(bm)
                                    break
                    else:  # exact mode (default)
                        benchmarks = list(
                            dict.fromkeys(self._only)
                        )  # remove duplicates while preserving order
                else:
                    benchmarks = find_enabled_benchmarks(
                        self.mode, REGISTERED_BENCHMARKS[self.name], self._skip
                    )

                # Handle prefix-with-baseline mode
                if (
                    self._only_match_mode == "prefix-with-baseline"
                    and self.name in BASELINE_BENCHMARKS
                ):
                    baseline_name = BASELINE_BENCHMARKS[self.name]
                    if baseline_name not in benchmarks:
                        benchmarks.append(baseline_name)

                # Run the baseline first, if baseline exists
                baseline_name = (
                    BASELINE_BENCHMARKS[self.name]
                    if self.name in BASELINE_BENCHMARKS
                    else None
                )
                if baseline_name and baseline_name in benchmarks:
                    benchmarks.remove(baseline_name)
                    benchmarks.insert(0, baseline_name)

                # get metrics for for each registered benchmark
                def _reduce_benchmarks(acc, bm_name: str):
                    baseline = (
                        bm_name == BASELINE_BENCHMARKS[self.name]
                        if self.name in BASELINE_BENCHMARKS
                        else False
                    )
                    acc[bm_name] = self._do_bench(
                        input_id=input_id,
                        fn_name=bm_name,
                        warmup=warmup,
                        rep=rep,
                        quantiles=quantiles,
                        baseline=baseline,
                    )
                    if baseline:
                        self.baseline_metrics = acc[bm_name]
                    return acc

                y_vals: Dict[str, BenchmarkOperatorMetrics] = functools.reduce(
                    _reduce_benchmarks, benchmarks, {}
                )
                metrics.append((x_val, y_vals))
                del self.example_inputs  # save some memory
                if "proton" in self.required_metrics:
                    proton.activate(self._proton_session_id)
                    proton.exit_scope()
                    proton.deactivate(self._proton_session_id)
            if "proton" in self.required_metrics:
                proton.activate(self._proton_session_id)
                proton.exit_scope()
                proton.finalize()
        except (KeyboardInterrupt, Exception):
            logger.warning(
                "Caught exception, terminating early with partial results",
                exc_info=True,
            )
            raise
        finally:
            self.output = BenchmarkOperatorResult(
                benchmark_name=self.tb_args.benchmark_name,
                op_name=self.name,
                op_mode=self.mode.value,
                metrics=self.required_metrics,
                simple_mode=self.tb_args.simple_output,
                result=metrics,
            )

    def get_x_val(self, example_inputs) -> Any:
        return self._cur_input_id

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        raise NotImplementedError(
            "Each operator must implement its own backward function."
        )

    def get_input_iter(self) -> Generator:
        """Return the dynamic input iterator for the model."""
        raise NotImplementedError(
            "Each operator must implement its own input iterator."
        )

    def get_grad_to_none(self, args):
        return None

    def plot(self):
        """Plot the comparison between different operator implementations."""
        raise NotImplementedError(
            "Each operator must implement its own plotting logic."
        )

    def best_config(self, fn):
        from unittest import mock

        from triton.runtime import Autotuner

        original_run = Autotuner.run
        autotuner = None

        def run_and_capture(self, *args, **kwargs):
            nonlocal autotuner
            autotuner = self
            original_run(self, *args, **kwargs)

        with mock.patch.object(Autotuner, "run", run_and_capture):
            fn()

        if autotuner is not None:
            return autotuner.best_config.all_kwargs()
        return None

    def all_configs(self, fn):
        from unittest import mock

        from triton.runtime import Autotuner

        from triton.runtime.jit import JITFunction

        original_run = Autotuner.run
        original_run_jit = JITFunction.run
        autotuner = None
        compiled_kernels = []

        def run_and_capture(self, *args, **kwargs):
            nonlocal autotuner
            autotuner = self
            original_run(self, *args, **kwargs)

        # There isn't really a great way to get the compiled kernels without monkeypatching
        def run_and_capture_jit(self, *args, **kwargs):
            compiled_kernel = original_run_jit(self, *args, **kwargs)
            compiled_kernels.append(compiled_kernel)
            return compiled_kernel

        with mock.patch.object(JITFunction, "run", run_and_capture_jit):
            with mock.patch.object(Autotuner, "run", run_and_capture):
                fn()

        if autotuner is not None and len(compiled_kernels):
            configs = []
            for config in autotuner.configs:
                configs.append(str(config))
            all_c = ",".join(configs)
            hashed = hashlib.sha256(all_c.encode("utf-8")).hexdigest()
            return hashed
        return None

    def kernel_hash(self, fn):
        try:
            AST = triton.compiler.ASTSource(fn=fn, signature={})
            sorted_sig = [v for k, v in sorted(AST.signature.items())]
            key = f"{AST.attrs.hash()}-{sorted_sig}"
            hashed = hashlib.sha256(key.encode("utf-8")).hexdigest()
            return hashed
        except:
            return ""

    def enable_bf16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.to(torch.bfloat16)
        self.dtype = torch.bfloat16
        self.example_inputs = input_cast(
            tensor_cond, tensor_action, self.example_inputs
        )

    def enable_fp16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.half()
        self.dtype = torch.float16
        self.example_inputs = input_cast(
            tensor_cond, tensor_action, self.example_inputs
        )

    # a function copied from https://fburl.com/code/hdypvhjw, which generate offsets
    # for jagged tensors with the given load_factor
    def generate_offsets(
        self,
        batch_size: int,
        max_seq_len: int,
        load_factor: float,
        offsets_dtype: torch.dtype,
    ) -> torch.Tensor:
        total_length = int(batch_size * max_seq_len * load_factor)
        avg_length = total_length // batch_size
        std = avg_length // 3  # rather arbitrary, but likely reasonable
        lengths = [random.gauss(avg_length, std) for _ in range(batch_size)]
        lengths = [int(min(max_seq_len, max(L, 0))) for L in lengths]

        if load_factor == 1.0:
            lengths = [max_seq_len] * batch_size

        diff = sum(lengths) - total_length
        idx_and_lengths = list(enumerate(lengths))
        random.shuffle(idx_and_lengths)

        for i, length in idx_and_lengths:
            if diff == 0:
                break
            elif diff > 0:
                delta = min(length, diff)
                lengths[i] -= delta
                diff -= delta
            else:
                delta = min(max_seq_len - length, -diff)
                lengths[i] += delta
                diff += delta

        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        return torch.tensor(
            offsets,
            dtype=offsets_dtype,
        )

    def enable_channels_last(self):
        tensor_cond = lambda x: x.dim() == 4
        tensor_action = lambda x: x.to(memory_format=torch.channels_last)
        self.example_inputs = input_cast(
            tensor_cond, tensor_action, self.example_inputs
        )

    def count_example_inputs(self):
        total_possible = sum(1 for _ in self.get_input_iter())
        if self._num_inputs is not None and self._num_inputs <= total_possible:
            return self._num_inputs
        return total_possible

    def get_example_inputs(self):
        if self._input_iter is None:
            self._input_iter = self.get_input_iter()
        try:
            return next(self._input_iter)
        except StopIteration:
            return None

    def get_temp_path(
        self,
        fn_name: Optional[str] = None,
    ) -> Path:
        unix_user: Optional[str] = os.environ.get("USER", None)
        logging_group: Optional[str] = self.logging_group
        parts = [x for x in ["tritonbench", unix_user, logging_group] if x]
        tritonbench_dir_name = "_".join(parts)
        benchmark_name = self.benchmark_name
        fn_part = f"{fn_name}_{self._input_id}" if fn_name else ""
        out_part = Path(tempfile.gettempdir()) / tritonbench_dir_name / benchmark_name
        return out_part / fn_part if fn_part else out_part

    @property
    def precision(self) -> str:
        if self.tb_args.precision == "bypass" or self.tb_args.precision == "fp8":
            return ""
        return self.tb_args.precision

    @property
    def benchmark_name(self, default: bool = False) -> str:
        if not default and self.tb_args.benchmark_name:
            return self.tb_args.benchmark_name
        parts = [x for x in [self.precision, self.name, self.mode.value] if x]
        return "_".join(parts)

    @property
    def logging_group(self) -> Optional[str]:
        return self.tb_args.logging_group

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        try:
            if self.mode == Mode.FWD:
                torch.testing.assert_close(
                    output,
                    baseline_output,
                    rtol=self.tb_args.rtol,
                    atol=self.tb_args.atol,
                )
            elif self.mode == Mode.BWD:
                torch.testing.assert_close(
                    output.grad,
                    baseline_output.grad,
                    rtol=self.tb_args.rtol,
                    atol=self.tb_args.atol,
                )
            else:
                fwd_output, loss = output
                baseline_fwd_output, baseline_loss = baseline_output
                torch.testing.assert_close(
                    fwd_output,
                    baseline_fwd_output,
                    rtol=self.tb_args.rtol,
                    atol=self.tb_args.atol,
                )
                torch.testing.assert_close(
                    loss.grad,
                    baseline_loss.grad,
                    rtol=self.tb_args.rtol,
                    atol=self.tb_args.atol,
                )
            return True
        except Exception:
            # either the output tensor or the loss grad tensor does not match
            return False

    def _do_bench(
        self,
        input_id: int,
        fn_name: str,
        warmup=DEFAULT_WARMUP,
        rep=DEFAULT_RUN_ITERS,
        quantiles=DEFAULT_QUANTILES,
        baseline: bool = False,
    ) -> BenchmarkOperatorMetrics:
        def _init_extra_metrics() -> Dict[str, Any]:
            extra_metrics = {}
            required_custom_metrics = set(REGISTERED_METRICS.get(self.name, [])) & set(
                self.required_metrics
            )
            for metric_name in required_custom_metrics:
                assert (
                    metric_name not in BUILTIN_METRICS
                ), "Metric name {metric_name} is built-in and should be OVERRIDDEN_METRICS. Please report a bug."
                extra_metrics[metric_name] = None
            return extra_metrics

        metrics = BenchmarkOperatorMetrics(
            hw_roofline=(
                self.hw_roofline() if "hw_roofline" in self.required_metrics else None
            ),
            extra_metrics=_init_extra_metrics(),
        )
        try:
            fn = self._get_bm_func(fn_name)
            if baseline:
                self.baseline_fn = fn
            if "cuda_time" in self.required_metrics:
                metrics.cuda_time = do_bench_cuda_time(
                    fn,
                    warmup,
                    rep,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                    use_cuda_graphs=self.use_cuda_graphs,
                    bypass_fail=self.tb_args.bypass_fail,
                )
            if {"latency", "tflops", "gbps", "speedup", "compile_time"} & set(
                self.required_metrics
            ):
                metrics.latency = do_bench_wrapper(
                    fn,
                    warmup,
                    rep,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                    device=self.device,
                    use_cuda_graphs=self.use_cuda_graphs,
                    bypass_fail=self.tb_args.bypass_fail,
                )
            if {
                "gpu_peak_mem",
                "gpu_mem_footprint_compression_ratio",
                "cpu_peak_mem",
            } & set(self.required_metrics):
                metrics.cpu_peak_mem, metrics.gpu_peak_mem = self.get_peak_mem(
                    fn,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                    required_metrics=self.required_metrics,
                    use_cuda_graphs=self.use_cuda_graphs,
                )
            if (
                "mem_footprint_compression_ratio" in self.required_metrics
                and "gpu_peak_mem" in self.required_metrics
                and self.baseline_metrics
            ):
                if (
                    self.baseline_metrics.gpu_peak_mem is not None
                    and metrics.gpu_peak_mem is not None
                ):
                    metrics.mem_footprint_compression_ratio = (
                        self.baseline_metrics.gpu_peak_mem / metrics.gpu_peak_mem
                    )
                else:
                    metrics.mem_footprint_compression_ratio = None
            if "walltime" in self.required_metrics:
                metrics.walltime = do_bench_walltime(
                    fn,
                    warmup=warmup,
                    rep=rep,
                )
            if "speedup" in self.required_metrics:
                metrics.speedup = (
                    self.baseline_metrics.latency / metrics.latency
                    if (self.baseline_metrics and self.baseline_metrics.latency)
                    and metrics.latency
                    else None
                )
                metrics.error_msg = (
                    self.baseline_metrics.error_msg
                    if self.baseline_metrics and self.baseline_metrics.error_msg
                    else None
                )
            if not baseline and "accuracy" in self.required_metrics:
                metrics.accuracy = (
                    self.accuracy(fn, self.baseline_fn) if self.baseline_fn else None
                )
            if "hw_roofline" in self.required_metrics:
                metrics.hw_roofline = self.hw_roofline()
            if "tflops" in self.required_metrics and metrics.latency:
                # cannot compute tflops without latency so adding latency to the check here
                metrics.tflops = self.tflops(fn_name, self.example_inputs, metrics)
            if "compile_time" in self.required_metrics:
                compile_time, compile_time_by_stage = self.compile_time(
                    input_id, fn_name, metrics
                )
                metrics.compile_time = compile_time
                if compile_time_by_stage:
                    metrics.compile_time_by_stage = compile_time_by_stage
            if "compile_trace" in self.required_metrics:
                metrics.compile_trace = self.compile_time(
                    input_id, fn_name, metrics, kineto_trace=True
                )
            if not is_hip():
                # ncu metrics (ncu_rep, ncu_rep_ir, or ncu_analyzer metrics)
                ncu_metrics: List[str] = ncu_analyzer.get_ncu_metrics(
                    self.required_metrics
                )
                if (
                    ncu_metrics
                    or "ncu_rep" in self.required_metrics
                    or "ncu_rep_ir" in self.required_metrics
                ):
                    profile_ir = "ncu_rep_ir" in self.required_metrics
                    out = self.ncu_trace(
                        input_id,
                        fn_name,
                        replay=True,
                        extend_ncu_args=ncu_metrics,
                        profile_ir=profile_ir,
                    )
                # Read and update NCU metrics if any required metrics match the NCU metrics
                if ncu_metrics:
                    ncu_analyzer_results = ncu_analyzer.read_ncu_report(
                        out, self.required_metrics
                    )
                    for metric_name, metric_value in ncu_analyzer_results.items():
                        metrics.extra_metrics[metric_name] = metric_value
                    if "arithmetic_intensity" in self.required_metrics:
                        logger.warning(
                            "Arithmetic intensity only supports FP32 and FP64 for now."
                        )
                if "ncu_rep" in self.required_metrics:
                    metrics.ncu_rep = out
                if "ncu_rep_ir" in self.required_metrics:
                    metrics.ncu_rep_ir = out
                # nsys metrics
                nsys_metrics = nsys_analyzer.get_nsys_metrics(self.required_metrics)
                if "nsys_rep" in self.required_metrics or nsys_metrics:
                    nsys_rep_path = self.nsys_rep(input_id, fn_name)
                    metrics.nsys_rep = nsys_rep_path
                    if nsys_metrics:
                        nsys_analyzer_results = nsys_analyzer.read_nsys_report(
                            nsys_rep_path, nsys_metrics
                        )
                        for metric_name, metric_value in nsys_analyzer_results.items():
                            metrics.extra_metrics[metric_name] = metric_value
                if "nsys_gpu_speedup" in self.required_metrics:
                    baseline_nsys_gpu_kernel_sum = (
                        self.baseline_metrics.extra_metrics.get(
                            "nsys_gpu_kernel_sum", None
                        )
                        if self.baseline_metrics
                        else None
                    )
                    current_nsys_gpu_kernel_sum = metrics.extra_metrics.get(
                        "nsys_gpu_kernel_sum", None
                    )
                    metrics.nsys_gpu_speedup = (
                        baseline_nsys_gpu_kernel_sum / current_nsys_gpu_kernel_sum
                        if baseline_nsys_gpu_kernel_sum and current_nsys_gpu_kernel_sum
                        else None
                    )
            else:
                if "att_trace" in self.required_metrics:
                    metrics.att_trace = self.att_trace(input_id, fn_name)
            if "kineto_trace" in self.required_metrics:
                metrics.kineto_trace = self.kineto_trace(input_id, fn)
            if "proton" in self.required_metrics:
                from tritonbench.components.proton import proton_trace

                scope_name = fn_name
                flops = self.flops() if self.has_metric("flops") else None
                num_bytes = self.bytes() if self.has_metric("bytes") else None
                proton_trace(
                    self._proton_session_id,
                    scope_name,
                    fn,
                    warmup=warmup,
                    flops=flops,
                    bytes=num_bytes,
                )
            if "best_config" in self.required_metrics:
                metrics.best_config = self.best_config(fn)
            if "all_configs" in self.required_metrics:
                metrics.all_configs = self.all_configs(fn)
            if "kernel_source_hash" in self.required_metrics:
                metrics.kernel_source_hash = self.kernel_hash(fn)
            if "_compile_time_kineto_trace_in_task" in self.required_metrics:
                assert (
                    self.required_metrics == ["_compile_time_kineto_trace_in_task"]
                    and len(self._only) == 1
                    and (self._input_id is not None)
                ), (
                    "_compile_time_kineto_trace_in_task must be measured by itself. "
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _input_id: {self._input_id}"
                )
                from tritonbench.components.compile_time import (
                    do_compile_kineto_trace_in_task,
                )

                kineto_trace_output_dir = self.get_temp_path(fn_name)
                kineto_trace_output_dir.mkdir(parents=True, exist_ok=True)
                metrics.extra_metrics["_compile_time_kineto_trace_in_task"] = (
                    do_compile_kineto_trace_in_task(
                        fn,
                        output_dir=str(kineto_trace_output_dir),
                        cold_start=self.tb_args.compile_cold_start,
                    )
                )
                self._compile_time_kineto_trace_in_task = metrics.extra_metrics[
                    "_compile_time_kineto_trace_in_task"
                ]
            if "_compile_time_in_task" in self.required_metrics:
                assert (
                    self.required_metrics == ["_compile_time_in_task"]
                    and len(self._only) == 1
                    and (self._input_id is not None)
                ), (
                    "_compile_time_in_task must be measured by itself. "
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _input_id: {self._input_id}"
                )
                from tritonbench.components.compile_time import do_compile_time_in_task

                if is_fbcode():
                    from tritonbench.components.compile_time import (
                        fbcode_do_compile_time_in_task,
                    )

                    compile_times = fbcode_do_compile_time_in_task(fn)
                    if compile_times is not None:
                        metrics.extra_metrics["compile_times"] = compile_times
                        self.compile_time_by_stage = {
                            k: v / 1_000_000
                            for k, v in compile_times.items()
                            if k != "total"
                        }
                        self.triton_hook_latency = (
                            compile_times["total"] / 1_000_000
                        )  # converting from ms to s
                if "compile_times" not in metrics.extra_metrics:
                    metrics.extra_metrics["_compile_time_in_task"] = (
                        do_compile_time_in_task(
                            fn, cold_start=self.tb_args.compile_cold_start
                        )
                    )
                    self._latency_with_compile_in_task = metrics.extra_metrics[
                        "_compile_time_in_task"
                    ]
            if "_ncu_trace_in_task" in self.required_metrics:
                assert (
                    self.required_metrics == ["_ncu_trace_in_task"]
                    and len(self._only) == 1
                    and (self._input_id is not None)
                ), (
                    "_ncu_trace_in_task must be measured by itself. "
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _input_id: {self._input_id}"
                )
                from tritonbench.components.ncu import do_bench_in_task

                do_bench_in_task(
                    fn=fn,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                    range_name=_RANGE_NAME,
                )
                metrics.extra_metrics["_ncu_trace_in_task"] = "success"
            if "_nsys_rep_in_task" in self.required_metrics:
                assert (
                    self.required_metrics == ["_nsys_rep_in_task"]
                    and len(self._only) == 1
                    and (self._input_id is not None)
                ), (
                    "_nsys_rep_in_task must be measured by itself. "
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _input_id: {self._input_id}"
                )
                from tritonbench.components.ncu import do_bench_in_task

                do_bench_in_task(
                    fn=fn,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                    range_name=_RANGE_NAME,
                    warmup=True,
                    use_cuda_profiler_range=True,
                )
                metrics.extra_metrics["_nsys_rep_in_task"] = "success"
            if self.tb_args.export:
                export_data(
                    x_val=self.get_x_val(self.example_inputs),
                    inputs=self.example_inputs,
                    fn_mode=self.mode.value,
                    fn=fn,
                    export_type=self.tb_args.export,
                    export_dir=self.tb_args.export_dir,
                )
            # generate customized metrics
            if self.name in REGISTERED_METRICS:
                for metric_name in REGISTERED_METRICS[self.name]:
                    if metric_name in BUILTIN_METRICS:
                        continue
                    if metric_name not in self.required_metrics:
                        continue
                    func = getattr(self, metric_name)
                    metrics.extra_metrics[metric_name] = func(
                        fn, self.example_inputs, metrics
                    )
            if self.tb_args.dump_ir:
                self.dump_ir(input_id, fn)
        except torch.cuda.OutOfMemoryError:
            metrics.error_msg = "CUDA OOM"
        except NotImplementedError:
            metrics.error_msg = "not supported"
        except Exception as e:
            if not self.tb_args.keep_going:
                raise
            metrics.error_msg = str(e)
        return metrics

    def do_bench_cudagraph_mem(
        self, fn, n_repeat=2, grad_to_none=None, device_type="cuda"
    ):
        with torch.cuda.stream(torch.cuda.Stream()):
            # warmup
            fn()
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.detach_()
                    x.requires_grad_(True)
                    x.grad = None
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(n_repeat):
                    if grad_to_none is not None:
                        for x in grad_to_none:
                            x.grad = None
                    fn()
            torch.cuda.synchronize()

    def do_bench_mem(self, fn, n_repeat=2, grad_to_none=None, device_type="cuda"):
        di = torch._dynamo.device_interface.get_interface_for_device(device_type)
        # warmup
        fn()
        di.synchronize()
        # benchmark
        for _ in range(n_repeat):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            fn()
        di.synchronize()

    def get_peak_mem(
        self,
        fn: Callable,
        grad_to_none: Optional[List[torch.Tensor]] = None,
        required_metrics: Optional[List[str]] = None,
        use_cuda_graphs: bool = False,
        device_type: str = "cuda",
    ) -> Tuple[Optional[float], Optional[float]]:
        """Measures peak CPU and GPU memory usage during function execution.

        Args:
            fn (Callable): The function to measure memory usage for.
            grad_to_none (Optional[List[torch.Tensor]], optional): List of tensors whose gradients
                should be set to None between iterations. Defaults to None.
            required_metrics (Optional[List[str]], optional): List of metrics to measure.
                Supported values: ["gpu_peak_mem", "mem_footprint_compression_ratio", "cpu_peak_mem"].
                Defaults to None.
            use_cuda_graphs (bool, optional): Whether to use CUDA graphs for measurement.
                Defaults to False.
            device_type (str, optional): Device to measure memory for ("cuda" or "cpu").
                Defaults to "cuda".

        Returns:
            Tuple[Optional[float], Optional[float]]: A tuple containing:
                - Peak CPU memory usage in GB (None if not requested)
                - Peak GPU memory usage in GB (None if not requested or not on CUDA)
        """
        gpu_peak_mem = None
        cpu_peak_mem = None
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        if use_cuda_graphs:
            self.do_bench_cudagraph_mem(
                fn, n_repeat=2, grad_to_none=grad_to_none, device_type=device_type
            )
        else:
            self.do_bench_mem(
                fn, n_repeat=2, grad_to_none=grad_to_none, device_type=device_type
            )
        if device_type == "cuda" and (
            {"gpu_peak_mem", "mem_footprint_compression_ratio"} & set(required_metrics)
        ):
            gpu_peak_mem = torch.cuda.max_memory_allocated() / 10**9
        if "cpu_peak_mem" in required_metrics:
            total = psutil.virtual_memory().total
            percentage = psutil.Process(os.getpid()).memory_percent()
            cpu_peak_mem = percentage * total / 10**9
        return cpu_peak_mem, gpu_peak_mem

    def _get_op_task_args(
        self, input_id: str, fn_name: str, in_task_metric_name: str
    ) -> List[str]:
        """
        Get arguments for running single operator backend and input in a subprocess.
        """
        op_task_args = [] if is_fbcode() else [sys.executable]
        op_task_args.extend(copy.deepcopy(sys.argv))
        op_task_args = remove_cmd_parameter(op_task_args, "--op")
        op_task_args = add_cmd_parameter(op_task_args, "--op", self.name)
        for override_option in ["--only", "--input-id", "--num-inputs", "--metrics"]:
            op_task_args = remove_cmd_parameter(op_task_args, override_option)
        op_task_args.extend(
            [
                "--only",
                fn_name,
                "--num-inputs",
                str(1),
                "--input-id",
                str(input_id),
                "--metrics",
                in_task_metric_name,
            ]
        )
        return op_task_args

    def nsys_rep(self, input_id: int, fn_name: str) -> str:
        op_task_args = self._get_op_task_args(input_id, fn_name, "_nsys_rep_in_task")
        nsys_output_dir = self.get_temp_path(fn_name)
        nsys_output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".nsys-rep"
        nsys_output_file = nsys_output_dir.joinpath(f"nsys_rep{ext}").resolve()
        nsys_trace_cmd = [
            "nsys",
            "profile",
            "-c",
            "cudaProfilerApi",
            "-t",
            "nvtx,osrt,cuda,cudnn,cublas",
            "-w",
            "true",
            "-f",
            "true",
            "-o",
            nsys_output_file,
        ]
        nsys_trace_cmd.extend(op_task_args)
        try:
            subprocess.check_call(nsys_trace_cmd)
        except subprocess.CalledProcessError:
            # FIXME: calling nsys on Tritonbench will throw SIGTERM with error code 143
            pass
        return str(nsys_output_file.resolve())

    def ncu_trace(
        self,
        input_id: int,
        fn_name: str,
        replay: bool = False,
        profile_ir=False,
        extend_ncu_args: List[str] = None,
    ) -> str:
        extend_ncu_args = (
            ["--metrics", ",".join(extend_ncu_args)]
            if extend_ncu_args
            else [
                "--set",
                "full",
            ]
        )
        op_task_args = self._get_op_task_args(input_id, fn_name, "_ncu_trace_in_task")
        # Disable DCGM
        disable_dyno_dcgm = [
            "sudo",
            "dyno",
            "dcgm_profiling",
            "--mute=true",
            "--duration=100000_s",
        ]
        disable_dcgm_service = [
            "sudo",
            "systemctl",
            "stop",
            "nvidia-dcgm",
        ]

        def service_exists(service_name):
            try:
                result = subprocess.run(
                    ["systemctl", "status", service_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                return result.returncode == 0
            except subprocess.CalledProcessError:
                return False

        if shutil.which("dyno") or service_exists("nvidia-dcgm"):
            dyno_result = subprocess.run(disable_dyno_dcgm).returncode
            systemctl_result = subprocess.run(disable_dcgm_service).returncode
            if dyno_result != 0 and systemctl_result != 0:
                logger.warn(
                    "DCGM may not have been successfully disabled. Proceeding to collect NCU trace anyway..."
                )
        ncu_output_dir = self.get_temp_path(fn_name)
        ncu_output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".csv" if not replay else ".ncu-rep"
        ncu_output_file = ncu_output_dir.joinpath(
            f"ncu_rep{'_ir' if profile_ir else ''}{ext}"
        ).resolve()
        ncu_args = [
            "ncu",
            "--nvtx",
            "--nvtx-include",
            # it is for range_start and range_end. no ending /.
            f"{_RANGE_NAME}",
            "--target-processes",
            "all",
            "--import-source",
            "yes",
        ]
        ncu_args.extend(extend_ncu_args)
        if replay:
            ncu_args.extend(
                [
                    "-f",
                    "-o",
                    str(ncu_output_file.resolve()),
                ]
            )
        else:
            ncu_args.extend(
                [
                    "--csv",
                    "-f",
                    "--log-file",
                    str(ncu_output_file.resolve()),
                ]
            )
        ncu_args.extend(op_task_args)
        logger.info("Running NCU: %s", shlex.join(ncu_args))
        # Sometimes, `ncu --target-processes all` will fail with the message "Failed to connect to process". Setting
        # CUDA_INJECTION64_PATH=none seems to fix this issue.
        env = {**os.environ, "CUDA_INJECTION64_PATH": "none"}
        if profile_ir:
            env["USE_TTGIR_LOC"] = "1"
        subprocess.check_call(ncu_args, env=env)
        return str(ncu_output_file.resolve())

    def att_trace(self, input_id: int, fn_name: str) -> str:
        op_task_args = self._get_op_task_args(input_id, fn_name, "_ncu_trace_in_task")
        att_output_dir = self.get_temp_path(fn_name)
        att_trace_dir = launch_att(att_output_dir, op_task_args)
        return att_trace_dir

    def kineto_trace(self, input_id: int, fn: Callable) -> str:
        from tritonbench.components.kineto import do_bench_kineto

        kineto_output_dir = self.get_temp_path(fn._name)
        kineto_output_dir.mkdir(parents=True, exist_ok=True)
        return do_bench_kineto(
            fn=fn,
            grad_to_none=self.get_grad_to_none(self.example_inputs),
            output_dir=kineto_output_dir,
            use_cuda_graphs=self.use_cuda_graphs,
        )

    def compile_time(
        self,
        input_id: int,
        fn_name: str,
        metrics: BenchmarkOperatorMetrics,
        kineto_trace: bool = False,
    ) -> Union[float, str]:
        # We need to spawn a subprocess when user wants to measure the compile time
        # of multiple sample inputs and backends.
        from tritonbench.operators.op_task import OpTask

        op_task_args = copy.deepcopy(self._raw_extra_args)
        for override_option in ["--only", "--input-id", "--num-inputs", "--metrics"]:
            op_task_args = remove_cmd_parameter(op_task_args, override_option)
        op_task_args.extend(
            [
                "--op",
                self.name,
                "--only",
                fn_name,
                "--num-inputs",
                str(1),
                "--input-id",
                str(input_id),
                "--metrics",
                (
                    "_compile_time_in_task"
                    if not kineto_trace
                    else "_compile_time_kineto_trace_in_task"
                ),
            ]
        )
        if self.tb_args.compile_cold_start:
            op_task_args.append("--compile-cold-start")
        # Set this env variable if you get a Permission denied issue
        # with local testing.
        output_dir = os.environ.get(
            "TRITONBENCH_COMPILE_TIME_OUTPUT_DIR", "/tmp/tritonbench"
        )
        op_task = OpTask(name=self.name, save_output_dir=Path(output_dir))
        op_task.make_operator_instance(args=op_task_args)
        op_task.run()
        if kineto_trace:
            kineto_trace_loc = op_task.get_attribute(
                "_compile_time_kineto_trace_in_task"
            )
            if is_fbcode():
                from tritonbench.components.kineto.fb.run_utils import (
                    manifold_upload_file,
                )

                return manifold_upload_file(kineto_trace_loc, perfdoctor=True)
            return kineto_trace_loc
        if op_task.get_attribute("triton_hook_latency") is not None:
            compiled_time = op_task.get_attribute("triton_hook_latency")
            compile_time_by_stage = op_task.get_attribute("compile_time_by_stage")
            return compiled_time, compile_time_by_stage
        latency_with_compile = op_task.get_attribute("_latency_with_compile_in_task")
        del op_task
        latency_without_compile = metrics.latency
        return latency_with_compile - latency_without_compile, None

    def hw_roofline(self) -> float:
        """Hardware roofline in tflops."""
        from tritonbench.utils.gpu_utils import HW_ROOFLINE_SPECS

        rooflines = HW_ROOFLINE_SPECS[self.is_compute_bound]

        device_name = (
            torch.cuda.get_device_name() if not torch.version.hip else "AMD MI300X"
        )
        assert (
            device_name in rooflines
        ), f"{device_name} is not supported in HW roofline specs."
        rooflines = rooflines[device_name]
        if self.is_compute_bound:
            assert (
                self.tb_args.precision in rooflines
            ), f"{self.tb_args.precision} is not supported by {device_name}."
            return rooflines[self.tb_args.precision]
        return rooflines

    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        if self.has_metric("flops"):
            flops = self.flops(fn_name, example_inputs, metrics)
            return flops / metrics.latency / 1e12 * 1e3

        def _get_flops(self, func: Callable) -> float:
            """By default, use the torch.__dispatch__ based flops counter."""
            from torch.utils.flop_counter import FlopCounterMode

            flop_counter = FlopCounterMode()

            def work_func():
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    func()
                    torch.cuda.synchronize()
                else:
                    func()

            with flop_counter:
                work_func()
            total_flops = flop_counter.get_total_flops()
            return total_flops

        fn = self._get_bm_func(fn_name)
        if fn not in self._op_flops:
            self._op_flops[fn] = _get_flops(self, fn)
        op_flops = self._op_flops[fn]
        return op_flops / metrics.latency / 1e12 * 1e3

    def dump_ir(self, input_id, fn):
        from unittest import mock

        from triton.runtime.jit import JITFunction

        original_run = JITFunction.run
        compiled_kernels = []

        # There isn't really a great way to get the compiled kernels without monkeypatching
        def run_and_capture(self, *args, **kwargs):
            compiled_kernel = original_run(self, *args, **kwargs)
            compiled_kernels.append(compiled_kernel)
            return compiled_kernel

        with mock.patch.object(JITFunction, "run", run_and_capture):
            fn()

        if len(compiled_kernels) > 0:
            ir_dir = self.get_temp_path(fn._name)
            ir_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Writing %s Triton IRs to %s",
                str(len(compiled_kernels)),
                ir_dir,
            )

        for kid, kernel in enumerate(compiled_kernels):
            for ir in ["ttir", "ttgir", "llir", "ptx", "amdgcn"]:
                if ir in kernel.asm:
                    with open(
                        ir_dir
                        / f"{fn._name}_{kernel.name}_k{kid}_input_x{input_id}.{ir}",
                        "w",
                    ) as f:
                        f.write(kernel.asm[ir])
            if "cubin" in kernel.asm:
                from triton.tools.disasm import get_sass

                sass = get_sass(kernel.asm["cubin"])
                with open(
                    ir_dir / f"{fn._name}_{kernel.name}_k{kid}_x{input_id}.sass", "w"
                ) as f:
                    f.write(sass)

    @classmethod
    def has_bwd(cls) -> bool:
        return cls.get_bwd_fn is not BenchmarkOperator.get_bwd_fn

    @classmethod
    def has_metric(cls, metric_name: str) -> bool:
        if metric_name == "tflops":
            return bool(getattr(cls, "flops", None))
        return bool(getattr(cls, metric_name, None))

    @classmethod
    def has_baseline(cls) -> Optional[str]:
        operator_name = cls.name
        return BASELINE_BENCHMARKS.get(operator_name, None)
