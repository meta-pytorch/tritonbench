import argparse

from tritonbench.utils.constants import DEFAULT_REP, DEFAULT_WARMUP

from tritonbench.utils.env_utils import AVAILABLE_PRECISIONS, is_fbcode


def get_parser(args=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--op",
        type=str,
        required=False,
        help="Operators to benchmark. Split with comma if multiple.",
    )
    parser.add_argument(
        "--op-collection",
        default="default",
        type=str,
        help="Operator collections to benchmark. Split with comma."
        " It is conflict with --op. Choices: [default, liger, all]",
    )
    parser.add_argument(
        "--mode",
        choices=["fwd", "bwd", "fwd_bwd", "fwd_no_grad"],
        default="fwd",
        help="Test mode (fwd, bwd, fwd_bwd, or fwd_no_grad).",
    )
    parser.add_argument("--bwd", action="store_true", help="Run backward pass.")
    parser.add_argument(
        "--fwd-bwd",
        action="store_true",
        help="Run both forward and backward pass.",
    )
    parser.add_argument(
        "--fwd-no-grad", action="store_true", help="Run forward pass without grad."
    )
    parser.add_argument(
        "--precision",
        "--dtype",
        choices=AVAILABLE_PRECISIONS,
        default="bypass",
        help="Specify operator input dtype/precision. Default to `bypass` - using DEFAULT_PRECISION defined in the operator.",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Num of warmup runs for each benchmark run.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=DEFAULT_REP,
        help="The rep time for each benchmark run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="The amount of time (in seconds) to sleep between benchmark runs.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print result as csv.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output result csv to the dir."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output result csv to file.",
    )
    parser.add_argument(
        "--output-json", type=str, default=None, help="Output result json to file."
    )
    parser.add_argument(
        "--skip-print",
        action="store_true",
        help="Skip printing result.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the result.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run in the CI mode.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Metrics to collect, split with comma. E.g., --metrics latency,tflops,speedup.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available metrics. Can be used with --op or --op-collection to show operator-specific metrics.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List all registerd backends per operator.",
    )
    parser.add_argument(
        "--metrics-gpu-backend",
        choices=["torch", "nvml"],
        default="torch",
        help=(
            "Specify the backend [torch, nvml] to collect metrics. In all modes, the latency "
            "(execution time) is always collected using `time.time_ns()`. The CPU peak memory "
            "usage is collected by `psutil.Process()`. In nvml mode, the GPU peak memory usage "
            "is collected by the `nvml` library. In torch mode, the GPU peak memory usage is "
            "collected by `torch.cuda.max_memory_allocated()`."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Specify one or multiple kernel implementations to run.",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Specify one or multiple kernel implementations to skip.",
    )
    parser.add_argument(
        "--only-match-mode",
        default="exact",
        choices=["exact", "prefix-with-baseline"],
        help="Match mode for --only argument. 'exact' for full string match, 'prefix-with-baseline' for prefix match as well as the existing baseline. Default: exact",
    )
    parser.add_argument(
        "--baseline", type=str, default=None, help="Override default baseline."
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        help="Number of example inputs.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
    )
    parser.add_argument(
        "--input-id",
        type=int,
        default=0,
        help="Specify the start input id to run. "
        "For example, --input-id 0 runs only the first available input sample."
        "When used together like --input-id <X> --num-inputs <Y>, start from the input id <X> "
        "and run <Y> different inputs.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run this under test mode, potentially skipping expensive steps like autotuning.",
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Dump Triton IR",
    )
    parser.add_argument(
        "--gpu-lockdown",
        action="store_true",
        help="Lock down GPU frequency and clocks to avoid throttling.",
    )
    parser.add_argument(
        "--operator-loader",
        action="store_true",
        help="Benchmarking aten ops in tritonbench/operator_loader.",
    )
    parser.add_argument(
        "--cudagraph", action="store_true", help="Benchmark with CUDA graph."
    )
    parser.add_argument(
        "--latency-measure-mode",
        default="triton_do_bench",
        choices=["triton_do_bench", "inductor_benchmarker"],
        help="Method to measure latency: triton_do_bench (default) or inductor_benchmarker.",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Run each operator in a separate child process. By default, it will always continue on failure.",
    )
    parser.add_argument(
        "--bypass-fail",
        action="store_true",
        help="bypass and continue on operator failure.",
    )
    parser.add_argument(
        "--shuffle-shapes",
        action="store_true",
        help="when true randomly shuffles the inputs before running benchmarks where possible.",
    )
    parser.add_argument(
        "--compile-cold-start",
        action="store_true",
        help="Include cold start time in compile_time and compile_trace.",
    )
    parser.add_argument(
        "--export",
        default=None,
        choices=["in", "out", "both"],
        help="Export input or output. Must be used together with --export-dir.",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        type=str,
        help="The directory to store input or output.",
    )
    parser.add_argument(
        "--benchmark-name",
        default=None,
        help="Name of the benchmark run.",
    )

    parser.add_argument(
        "--prod-shapes",
        action="store_true",
        help="Only run with pre-defined production shapes.",
    )
    parser.add_argument(
        "--simple-output",
        action="store_true",
        help="Only print the simple output.",
    )

    parser.add_argument(
        "--tritonparse",
        nargs="?",
        const="./tritonparse_logs/",
        default=None,
        help="Enable tritonparse structured logging. Optionally specify log directory path (default: ./tritonparse_logs/).",
    )
    parser.add_argument(
        "--input-loader",
        type=str,
        help="Load input file from Tritonbench data JSON.",
    )
    parser.add_argument(
        "--logging-group",
        type=str,
        default=None,
        help="Name of group for benchmarking.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance for accuracy metric.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance for accuracy metric.",
    )

    # A/B Testing parameters
    parser.add_argument(
        "--side-a",
        type=str,
        default=None,
        help="Configuration A for A/B testing. Specify operator-specific arguments as a string. "
        "Example: '--side-a \"--max-autotune --dynamic\"'",
    )
    parser.add_argument(
        "--side-b",
        type=str,
        default=None,
        help="Configuration B for A/B testing. Specify operator-specific arguments as a string. "
        "Example: '--side-b \"--dynamic\"'",
    )

    if is_fbcode():
        parser.add_argument("--log-scuba", action="store_true", help="Log to scuba.")
        parser.add_argument(
            "--production-shapes",
            action="store_true",
            help="whether or not to take specific production shapes as input",
        )
        parser.add_argument(
            "--triton-type",
            default="stable",
            type=str,
            help="Set what version of Triton we are using for logging purposes.",
        )

    args, extra_args = parser.parse_known_args(args)
    if args.op and args.ci:
        parser.error("cannot specify operator when in CI mode")
    if not args.op and not args.op_collection:
        print(
            "Neither operator nor operator collection is specified. Running all operators in the default collection."
        )

    # A/B Testing validation
    if (args.side_a is not None) != (args.side_b is not None):
        parser.error(
            "A/B testing requires both --side-a and --side-b arguments to be specified together"
        )

    if args.side_a is not None and args.side_b is not None:
        # A/B mode is enabled
        if not args.op:
            parser.error(
                "A/B testing requires a specific operator (--op) to be specified"
            )
        if args.op_collection != "default":
            parser.error(
                "A/B testing is only supported with single operators, not operator collections"
            )
        if "," in args.op:
            parser.error(
                "A/B testing is only supported with a single operator, not multiple operators"
            )
        if args.isolate:
            parser.error("A/B testing is not compatible with --isolate mode")
    return parser
