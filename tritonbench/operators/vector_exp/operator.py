from typing import Any, Callable, Generator, List

import torch
import triton

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import triton_exp_kernel


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "gbps"]

    @register_metric()
    def gbps(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics):
        def normalize(lat):
            return (
                3
                * example_inputs[0].element_size()
                * example_inputs[0].numel()
                / lat
                * 1e-6
            )

        return (
            normalize(metrics.latency),
            normalize(metrics.latency.max),
            normalize(metrics.latency.min),
        )

    @register_metric()
    def duration(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics):
        output = fn()
        if output is None:
            return None
        return (
            torch.mean(output, dtype=torch.float32).item(),
            torch.max(output).item(),
            torch.min(output).item(),
        )

    @register_benchmark()
    def triton_exp(self, x: torch.Tensor):
        # We need to preallocate the output.
        output = torch.empty_like(x)
        n_elements = output.numel()
        # The SPMD launch grid denotes the number of kernel instances that run in parallel.
        # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
        # In this case, we use a 1D grid where the size is the number of blocks:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        # NOTE:
        #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
        #  - Don't forget to pass meta-parameters as keywords arguments.

        # Prepare a memory buffer to store the profiled data, with the size equal to the number of programs.
        BLOCK_SIZE = 1024
        n_programs = triton.cdiv(n_elements, BLOCK_SIZE)
        profile_mem = torch.empty(n_programs, dtype=torch.int64, device=self.device)

        def _inner():
            triton_exp_kernel[grid](
                x, output, n_elements, BLOCK_SIZE=1024, profile_mem=profile_mem
            )
            return {"output": output, "profile_mem": profile_mem}

        return _inner

    @register_benchmark(baseline=True)
    def torch_exp(self, x: torch.Tensor):
        def _inner():
            output = torch.exp(x)
            return {"output": output}

        return _inner

    @register_benchmark()
    def torch_compile_exp(self, x: torch.Tensor):
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def _compiled_exp(x):
            return torch.exp(x)
        
        def _inner():
            output = _compiled_exp(x)
            return {"output": output}
        
        return _inner

    def get_x_vals(self) -> List[int]:
        return [2**i for i in range(12, 28, 1)]

    def get_x_val(self, example_inputs):
        return len(example_inputs[0])

    @register_metric()
    def accuracy(self, fn, baseline_fn) -> bool:
        output = fn()["output"]
        baseline_output = baseline_fn()["output"]
        try:
            torch.allclose(output, baseline_output)
            return True
        except Exception:
            return False

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["size"],  # Argument names to use as an x-axis for the plot.
                x_vals=self.output.x_vals,  # Different possible values for `x_name`.
                x_log=True,  # x axis is logarithmic.
                line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
                line_vals=[
                    "torch_exp",
                    "triton_exp",
                ],  # Possible values for `line_arg`.
                line_names=["Torch", "Triton"],  # Label name for the lines.
                styles=[("blue", "-"), ("green", "-")],  # Line styles.
                ylabel="GB/s",  # Label name for the y-axis.
                plot_name="vector-exp-performance",  # Name for the plot. Used also as a file name for saving the plot.
                args={},  # Values for function arguments not in `x_names` and `y_name`.
            )
        )
        def _plot(size, provider):
            gbps, max_gbps, min_gbps = self.output.get_y_vals(size, provider, "gbps")
            return gbps, max_gbps, min_gbps

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/vector_exp")

    def get_input_iter(self) -> Generator:
        for size in self.get_x_vals():
            x = torch.rand(size, device=self.device, dtype=self.dtype)
            yield (x,)
