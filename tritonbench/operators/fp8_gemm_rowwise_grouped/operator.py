"""
TorchBench FP8 Grouped GEMM Rowwise Operator Benchmark
=============================================

This module provides a benchmark for the FP8 GEMM grouped operator using TorchBench.

Description
-----------

The FP8 GEMM grouped operator is a key component in many deep learning models, particularly those involving matrix multiplications. This benchmark evaluates the performance of different implementations of this operator, including Triton, Cutlass, and CK.

Usage
-----

To use this benchmark, simply run the buck cmd with the desired command-line arguments. The available arguments are:

* `--m`: The number of rows in the input matrix.
* `--n`: The number of columns in the input matrix.
* `--k`: The number of columns in the weight matrix.
* `--group_size`: The size of the groups in the grouped GEMM operation.
* `--llama`: TODO: Whether to use the LLaMA model shapes.
* `--prefill`: TODO: Whether to use prefill shapes.
* `--no_fp8_fast_accum`: Whether to disable fast accumulation for FP8.
* `--no_use_tma`: TODO: Whether to disable the use of TMA (Tensor Memory Accelerator).
* `--use_tma`: TODO: Whether to enable the use of TMA.
* `--no_use_persistent`: TODO: Whether to disable the use of persistent memory.
* `--warp_specialization`: Whether to enable warp specialization.

Example usage:
buck2 run -c fbcode.platform010_cuda_version=12.4  @mode/opt //pytorch/tritonbench:run -- --op fp8_gemm_rowwise_grouped

Functions/Classes
-----------------
* `Operator`: A class representing the FP8 GEMM grouped operator benchmark.
* `_triton`: A function implementing the Triton version of the FP8 GEMM grouped operator.
* `_cutlass_or_ck`: A function implementing the Cutlass or CK version of the FP8 GEMM grouped operator.
* `_torch`: TODO: A function implementing the Torch version of the FP8 GEMM grouped operator.
* `get_input_iter`: A generator function providing input data for the benchmark.
* `plot`: A function plotting the performance results of the benchmark.

Metrics
-------
The following metrics are measured by this benchmark:
* TFLOPS (tera floating-point operations per second)
* GB/s (gigabytes per second)

"""

# Import necessary libraries and modules
import argparse
import random
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
import triton
from tritonbench.utils.data_utils import get_production_shapes

from tritonbench.utils.env_utils import is_fbcode

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    gemm_shapes,
    register_benchmark,
    register_metric,
    register_x_val,
)


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Parse command-line arguments for the TorchBench FP8 GEMM grouped operator benchmark.

    Args:
        args (List[str]): A list of command-line arguments.

    Returns:
        argparse.Namespace: A namespace containing the parsed arguments.

    Note: Many of the flags are currently unsupported & will be added incrementally.
    """

    # Create an ArgumentParser instance to define the command-line arguments
    parser = argparse.ArgumentParser(
        description="TorchBench fp8 gemm grouped operator Benchmark"
    )

    # Define the command-line arguments
    parser.add_argument("--m", type=int, help="The number of rows in the input matrix.")
    parser.add_argument(
        "--n", type=int, help="The number of columns in the input matrix."
    )
    parser.add_argument(
        "--k", type=int, help="The number of columns in the weight matrix."
    )
    parser.add_argument(
        "--group_size",
        type=int,
        help="The size of the groups in the grouped GEMM operation.",
    )
    parser.add_argument("--llama", action="store_true", help="Use LLaMA model shapes.")
    parser.add_argument(
        "--prefill", default=False, action="store_true", help="Use prefill shapes."
    )
    parser.add_argument(
        "--no_fp8_fast_accum",
        dest="fp8_fast_accum",
        action="store_false",
        help="Disable fast accumulation for FP8.",
    )
    parser.add_argument(
        "--no_use_tma",
        dest="use_tma",
        default=None,
        action="store_false",
        help="Disable the use of TMA (Tensor Memory Accelerator).",
    )
    parser.add_argument(
        "--use_tma",
        dest="use_tma",
        action="store_true",
        help="Enable the use of TMA (Tensor Memory Accelerator).",
    )
    parser.add_argument(
        "--no_use_persistent",
        dest="no_use_persistent",
        action="store_true",
        help="Disable the use of persistent memory.",
    )
    parser.add_argument(
        "--warp_specialization",
        action="store_true",
        help="Enable warp specialization.",
    )

    # Parse the command-line arguments
    parsed_args = parser.parse_args(args)

    # Set default values for certain arguments based on the platform
    if parsed_args.use_tma is None:
        # Default to True for CUDA, False for ROCm
        parsed_args.use_tma = True if torch.version.hip is None else False

    # Check for incompatible arguments on ROCm platform
    if torch.version.hip is not None:
        if parsed_args.use_tma:
            raise RuntimeError("TMA is not supported on ROCm")
        parsed_args.no_use_persistent = True  # default true for ROCm
        if parsed_args.warp_specialization:
            parsed_args.warp_specialization = False
            print("Warp specialization is not supported on ROCm defaulting to False")

    return parsed_args


# Define flags to track the availability of different kernels
HAS_CUBLAS = False  # TODO: add cublas kernel
HAS_TRITON = False
HAS_CUTLASS_OR_CK = False

from tritonbench.utils.fp8_utils import get_fp8_constants

# Try to import Triton grouped GEMM module
try:
    from mslk.gemm.triton.grouped_gemm import grouped_gemm_fp8_rowwise

    # If import succeeds, set HAS_TRITON to True
    HAS_TRITON = True
except (ImportError, AssertionError):
    # If import fails, set HAS_TRITON to False
    HAS_TRITON = False

# Try to import Cutlass or CK module
try:
    import mslk.gemm  # noqa: F401

    # Define the Cutlass or CK FP8 grouped MM operator
    cutlass_or_ck_fp8_grouped_mm = torch.ops.mslk.f8f8bf16_rowwise_grouped_stacked
    # Set HAS_CUTLASS_OR_CK to True if import succeeds
    HAS_CUTLASS_OR_CK = True
except (ImportError, AttributeError, OSError):
    # Set HAS_CUTLASS_OR_CK to False if import fails
    HAS_CUTLASS_OR_CK = False


BUILTIN_SHAPES = [
    # (128, 128, 128),
    # (256, 256, 256),
    # (512, 512, 512),
    # (2048, 2048, 2048),
    (1024, 1024, 1024),
    (2048, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 4096, 4096),
    # (16384, 4096, 4096),
    # (8192, 8192, 8192),
    # (16384, 8192, 8192),
    # (16384, 16384, 16384),
    # (1, 2304, 2048),
    # (1, 8192, 16384),
    # (4, 4096, 2304),
    # (4, 13312, 2048),
    # (8, 2304, 2304),
    # (8, 8192, 6656),
    # (16, 4096, 6656),
    # (16, 13312, 13312),
    # (32, 2304, 16384),
    # (32, 8192, 13312),
    # (64, 4096, 2048),
    # (64, 13312, 2048),
    # (128, 2304, 6656),
    # (128, 8192, 2304),
    # (2048, 8192, 2048),
    # (2048, 13312, 6656),
    # (4096, 2304, 13312),
    # (4096, 13312, 2304),
    # (16384, 4096, 16384),
    # (16384, 8192, 13312),
]

GROUP_SIZES = [
    2,
    4,
]  # 8, 16]

FP8_DTYPE, _, _, _ = get_fp8_constants()
E4M3_MAX_POS: float = torch.finfo(FP8_DTYPE).max
EPS: float = 1e-12
FP16_MAX_POS: float = torch.finfo(torch.float16).max


def fp8_row_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize an input tensor to FP8 format.

    This function calculates the maximum absolute value of each row in the input tensor,
    then scales the tensor by the reciprocal of this maximum value. The scaled tensor is
    then clamped to the range [-E4M3_MAX_POS, E4M3_MAX_POS] and converted to FP8 format.

    Args:
        x (torch.Tensor): The input tensor to be quantized.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the quantized tensor and its inverse scale.
    """

    # Calculate the maximum absolute value of each row in the input tensor
    row_max = torch.max(torch.abs(x), dim=1).values

    # Calculate the scale factor for each row
    scale = E4M3_MAX_POS / torch.clamp(row_max, EPS)

    # If the input tensor is in float16 format, clamp the scale factor to prevent overflow
    if x.dtype is torch.float16:
        scale = torch.clamp(scale, max=FP16_MAX_POS)

    # Scale the input tensor and clamp it to the range [-E4M3_MAX_POS, E4M3_MAX_POS]
    xq = torch.clamp(x * scale[:, None], min=-E4M3_MAX_POS, max=E4M3_MAX_POS)

    # Convert the scaled tensor to FP8 format
    xq = xq.to(FP8_DTYPE)

    # Return the quantized tensor and its inverse scale
    return xq, scale.reciprocal().to(torch.float32)


def cumulative_sum_with_initial_offset(tensor):
    """
    Calculate the cumulative sum of a 1D tensor with an initial offset of 0.
    Args:
        tensor (torch.Tensor): A 1D tensor.
    Returns:
        torch.Tensor: The cumulative sum of the input tensor with an initial offset of 0.
    """
    cumsum = torch.zeros_like(tensor)
    cumsum[1:] = torch.cumsum(tensor[:-1], dim=0)
    return cumsum


def reshape_tensor(input_tensor, m_sizes):
    """
    Reshape the input tensor into a specified grouped format.
    This function takes an input tensor and reshapes it into a 3D tensor
    with dimensions (G, N, K), where:
    - G is the number of groups, determined by the length of m_sizes.
    - N is the size of each group, calculated as the integer division of
      the first dimension of the input tensor by G.
    - K is the size of the second dimension of the input tensor.
    Args:
        input_tensor (torch.Tensor): The input tensor to be reshaped. It is
            expected to have at least two dimensions.
        m_sizes (list): A list whose length determines the number of groups (G).
    Returns:
        torch.Tensor: The reshaped tensor with dimensions (G, N, K).
    Raises:
        ValueError: If the size of the first dimension of input_tensor is not
            divisible by the number of groups (G).
    """
    # Calculate the number of groups (G) based on the length of m_sizes
    G = len(m_sizes)

    # Calculate the size of each group (N) by dividing the first dimension of
    # the input tensor by the number of groups (G)
    N = input_tensor.size(0) // G

    # Get the size of the second dimension (K) of the input tensor
    K = input_tensor.size(1)
    # Reshape the input tensor to have dimensions (G, N, K)
    reshaped_tensor = input_tensor.view(G, N, K)
    return reshaped_tensor


class Operator(BenchmarkOperator):
    """
    A benchmark operator class for FP8 GEMM grouped operations.
    This class provides a set of methods to benchmark and compare the performance of different
    implementations of FP8 GEMM grouped operations, including Triton, Cutlass, and CK.

    Methods:
        __init__: Initializes the operator with the given arguments.
        _triton: A benchmark function for the Triton implementation.
        _cutlass_or_ck: A benchmark function for the Cutlass or CK implementation.
        get_x_val: Returns the x-value for the benchmark plot.
        flops: Calculates the FLOPS metric for the benchmark.
        gbps: Calculates the GB/s metric for the benchmark.
        get_input_iter: Returns an iterator over the input data for the benchmark.
        _get_accuracy: Checks the accuracy of the benchmark results.
        plot: Plots the benchmark results.
    """

    DEFAULT_METRICS = ["tflops", "gbps", "speedup", "accuracy"]
    DEFAULT_PRECISION = "fp8"
    FWD_ONLY = True

    def __init__(
        self,
        tb_args: argparse.Namespace,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the Operator instance.

        Args:
            tb_args (argparse.Namespace): The parsed command-line arguments.
            extra_args (Optional[List[str]], optional): Additional command-line arguments. Defaults to None.

        Notes:
            This method initializes the Operator instance by parsing the command-line arguments,
            setting up the shapes and group sizes for the benchmark, and configuring other parameters.
        """

        # Call the parent class's __init__ method to initialize the base attributes
        super().__init__(tb_args, extra_args)

        # With `--latency-measure-mode profiler`, we no longer need CUDA graphs for accurate latency measurement.
        # self.use_cuda_graphs = True

        # Enable fp8_fast_accum by default. The cutlass kernel does not support configuring
        # this parameter as of now. By default it is true, but there will be correctness issues
        # vs the cutlass kernel, if fp8_fast_accum is turned off.
        self.fp8_fast_accum = True

        # Parse the additional command-line arguments
        addmm_args = parse_args(self.extra_args)

        # Determine the shapes for the benchmark based on the command-line arguments
        if addmm_args.m and addmm_args.n and addmm_args.k:
            # Use the specified shapes if provided
            self.shapes = [(addmm_args.m, addmm_args.n, addmm_args.k)]
        elif addmm_args.llama:
            # Use the LLaMA shapes if specified
            self.shapes = gemm_shapes(addmm_args.prefill)
        else:
            # Use the built-in shapes otherwise
            self.shapes = BUILTIN_SHAPES

        # Determine the group sizes for the benchmark based on the command-line arguments
        if addmm_args.group_size:
            # Use the specified group size if provided
            self.group_sizes = [addmm_args.group_size]
        else:
            # Use the default group sizes otherwise
            self.group_sizes = GROUP_SIZES

        # Configure other parameters based on the command-line arguments
        self.fp8_fast_accum = addmm_args.fp8_fast_accum
        self.use_tma = addmm_args.use_tma
        self.no_use_persistent = addmm_args.no_use_persistent
        self.warp_specialization = addmm_args.warp_specialization

    @register_benchmark(enabled=HAS_TRITON)
    def _triton(self, group_A, group_B, m_sizes, a_scale, b_scale) -> Callable:
        """
        Returns a lambda function that performs the Triton FP8 GEMM grouped operation.

        Args:
            group_A (torch.Tensor): The first input tensor.
            group_B (torch.Tensor): The second input tensor.
            m_sizes (List[int]): The sizes of the groups.
            a_scale (torch.Tensor): The scale factor for the first input tensor.
            b_scale (torch.Tensor): The scale factor for the second input tensor.

        Returns:
            Callable: A lambda function that performs the Triton FP8 GEMM grouped operation.
        """
        # Return a lambda function that calls the grouped_gemm_fp8_rowwise function
        return lambda: grouped_gemm_fp8_rowwise(
            group_A,
            group_B,
            m_sizes,
            a_scale,
            b_scale,
            use_fast_accum=self.fp8_fast_accum,
            _use_warp_specialization=self.warp_specialization,
        )

    @register_benchmark(
        enabled=HAS_CUTLASS_OR_CK and is_fbcode(),
        label="ck" if torch.version.hip else "cutlass",
    )
    def _cutlass_or_ck(self, group_A, group_B, m_sizes, a_scale, b_scale) -> Callable:
        """
        Returns a lambda function that performs the Cutlass or CK FP8 GEMM grouped operation.

        Args:
            group_A (torch.Tensor): The first input tensor.
            group_B (torch.Tensor): The second input tensor.
            m_sizes (List[int]): The sizes of the groups.
            a_scale (torch.Tensor): The scale factor for the first input tensor.
            b_scale (torch.Tensor): The scale factor for the second input tensor.

        Returns:
            Callable: A lambda function that performs the Cutlass or CK FP8 GEMM grouped operation.
        """
        # Reshape group_B to match the format expected by the cutlass implementation (G, N, K)
        reshaped_group_B = reshape_tensor(group_B, m_sizes)

        # Return a lambda function that calls the cutlass_or_ck_fp8_grouped_mm function
        return lambda: cutlass_or_ck_fp8_grouped_mm(
            group_A,
            reshaped_group_B,
            a_scale,
            b_scale,
            m_sizes.to(torch.int64),
        )

    @register_x_val(label="(group_size, M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int, int]:
        """
        Returns the x-value for the benchmark plot.

        The x-value is a tuple containing the group size, matrix dimensions M, N, and K.

        Args:
            example_inputs: A tuple of input tensors and scales.

        Returns:
            Tuple[int, int, int, int]: A tuple containing the group size, matrix dimensions M, N, and K.
        """

        # Unpack the example inputs into individual variables
        group_A, group_B, m_sizes, a_scale, b_scale = example_inputs

        # Calculate the group size from the length of the m_sizes list
        group_size = len(m_sizes)

        # Extract the matrix dimensions from the input tensors
        xq, wq = group_A, group_B
        m, k = xq.size()
        gn, k = wq.size()

        # Calculate the value of N by dividing the total number of columns by the group size
        n = gn // group_size

        # Return the x-value as a tuple
        return (group_size, m, n, k)

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        """
        Calculate the floating point operations per second (FLOPS) for a given operation.
        Args:
            fn_name (str): The name of the function being benchmarked.
            example_inputs (Any): A tuple containing the inputs to the function.
            metrics (BenchmarkOperatorMetrics): An object containing metrics about the benchmark.
        Returns:
            float: A list containing the FLOPS value.
        """
        # Unpack the example inputs
        group_A, group_B, m_sizes, a_scale, b_scale = example_inputs
        # Get the sizes of the input tensors
        xq, wq = group_A, group_B
        m, k = xq.size()  # input size
        gn, k = wq.size()  # weight size
        group_size = len(m_sizes)
        n = gn // group_size
        # Calculate the FLOPS
        flops = n * m * k * 2
        return flops

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:
        """
        Calculate the memory bandwidth in GB/s for a given operation.
        Args:
            fn: The function being benchmarked.
            example_inputs (Any): A tuple containing the inputs to the function.
            metrics (BenchmarkOperatorMetrics): An object containing metrics about the benchmark.
        Returns:
            float: The memory bandwidth in GB/s.
        """

        def nbytes(t):
            """
            Calculate the number of bytes occupied by a tensor.
            Args:
                t: The tensor.
            Returns:
                int: The number of bytes.
            """
            return t.numel() * t.element_size()

        # Unpack the example inputs
        group_A, group_B, m_sizes, a_scale, b_scale = example_inputs
        # Run the function and get the output
        c = fn()
        # If the output is a tuple, extract the first element
        c = c[0] if isinstance(c, tuple) else c
        # Calculate the total memory used in GB
        gb = (nbytes(group_A) + nbytes(group_B) + nbytes(c)) / 1e9
        # Calculate the memory bandwidth in GB/s
        gbps = gb / metrics.latency * 1e3
        return gbps

    def get_input_iter(self) -> Generator:
        """
        Generate input tensors for a matrix multiplication operation.

        Yields:
            tuple: A tuple containing the input tensors and their corresponding scales.
        """

        # Iterate over all possible group sizes and shapes
        for group_size in self.group_sizes:
            for shape in self.shapes:
                # Unpack the shape into its dimensions
                m, n, k = shape

                # Create a random tensor B with the specified shape and data type
                B = torch.randn(
                    (group_size * n, k), device=self.device, dtype=torch.bfloat16
                ).requires_grad_(False)

                # Quantize tensor B to FP8 format
                group_B, b_scale = fp8_row_quantize(B)

                # Calculate the size of each group in the m dimension
                m_sizes = [m // group_size] * group_size

                # Convert the list of group sizes to a tensor
                m_sizes = torch.tensor(m_sizes, device=self.device, dtype=torch.int32)

                # Create a random tensor A with the specified shape and data type
                A = torch.randn(
                    (m, k), device=self.device, dtype=torch.bfloat16
                ).requires_grad_(False)

                # Quantize tensor A to FP8 format
                group_A, a_scale = fp8_row_quantize(A)

                # Yield the quantized tensors and their corresponding scales
                yield group_A, group_B, m_sizes, a_scale, b_scale

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        """
        Check if the output of a function matches the output of a baseline function.
        Args:
            fn (Callable): The function to check.
            baseline_fn (Callable): The baseline function to compare against.
        Returns:
            bool: True if the outputs match within a tolerance, False otherwise.
        """
        # Run the function and get its output
        output = fn()
        # Run the baseline function and get its output
        baseline_output = baseline_fn()
        try:
            # Compare the outputs using PyTorch's assert_close function
            torch.testing.assert_close(output, baseline_output, atol=1e-2, rtol=0.5)
            # If no exception is raised, the outputs match
            return True
        except Exception:
            # If an exception is raised, the outputs do not match
            return False

    def plot(self):
        """
        Generate a performance plot for the GEMM operation.

        The plot shows the TFLOPS achieved by different providers (Triton, CK, Cutlass)
        for various densities.
        """

        # Define the benchmark configuration
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["density"],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "_triton",
                    "_ck" if torch.version.hip else "_cutlass",
                ],  # possible values for `line_arg``
                line_names=[
                    "Triton",
                    "CK" if torch.version.hip else "Cutlass",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-"), ("yellow", "-")],  # line styles
                ylabel="tflops",  # label name for the y-axis
                plot_name="gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(density, provider):
            """
            Get the TFLOPS value for a given density and provider.

            Args:
                density: The density value.
                provider: The provider name.

            Returns:
                float: The TFLOPS value.
            """
            tflops = self.output.get_y_vals(density, provider, "tflops")
            return tflops

        # Set the save path for the plot
        save_path = "/tmp/test_fp8_gemm_grouped"

        # Run the plot and save it to the specified path
        _plot.run(show_plots=True, print_data=True, save_path=save_path)

    @register_benchmark(baseline=True)
    def eager_fp8_gemm_rowwise_grouped(
        self, group_A, group_B, m_sizes, a_scale, b_scale
    ) -> Callable:
        def torch_perf_fn(group_A, group_B, m_sizes, a_scale, b_scale):
            group_size = len(m_sizes)
            xq, wq = group_A, group_B
            m, k = xq.size()
            gn, _ = wq.size()
            n = gn // group_size

            expected_result = torch.zeros(
                m, n, dtype=torch.bfloat16, device=self.device
            )

            # m_sizes holds the row count for each group; cumulative offsets mark
            # the starting row for every group in the flattened input tensor.
            m_starts = cumulative_sum_with_initial_offset(m_sizes)

            for g in range(group_size):
                m_start = int(m_starts[g].item())
                m_end = m_start + int(m_sizes[g].item())
                n_start = g * n
                n_end = n_start + n

                # Dequantize the FP8 tiles with their row-wise scales before performing
                # the per-group matmul against the matching weight slice.
                expected_result[m_start:m_end, :] = (
                    group_A[m_start:m_end, :].to(torch.float32)
                    @ group_B[n_start:n_end, :].to(torch.float32).T
                    * a_scale[m_start:m_end][:, None]
                    * b_scale[n_start:n_end][None, :]
                ).to(torch.bfloat16)

            return expected_result

        return lambda: torch_perf_fn(group_A, group_B, m_sizes, a_scale, b_scale)
