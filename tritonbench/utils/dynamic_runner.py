from typing import Any, Callable, Dict, Generator, List, Optional

from tritonbench.utils.triton_op import BenchmarkOperator, BenchmarkOperatorResult


def dynamic_run(
    benchmarks: Dict[str, Callable],
    input_iter: Optional[Generator],
    silent=False,
    **kwargs: Any,
) -> BenchmarkOperatorResult:
    """
    Run a list of benchmarks with a given set of inputs and kwargs.
    Kwargs in this case are the command-line arguments available in tritonbench.

    Example:

    def triton_add(x, y):
        ...

    def input_iter():
        size = 2**12
        x = torch.rand(size, device="cuda")
        y = torch.rand(size, device="cuda")
        yield x,y

    benchmarks = {
        "triton_add": triton_add,
        "triton_add2": triton_add,
    }

    dynamic_run(benchmarks=benchmarks, input_iter=input_iter, benchmark_name="vector_add")
    """

    # Convert kwargs into a list of command-line arguments
    arg_list = []
    for k, v in kwargs.items():
        key = f"--{k.replace('_', '-')}"
        arg_list.append(key)
        arg_list.append(str(v))

    op = BenchmarkOperator(extra_args=arg_list)

    op.set_input_iter(input_iter)

    for k, v in benchmarks.items():
        op.add_benchmark(bm_func_name=k, bm_callable=v)

    op.run()
    if not silent:
        print(op.output)
    return op.output


def dynamic_run_once(
    benchmarks: Dict[str, Callable],
    single_input: Optional[List[Any]],
    silent=False,
    **kwargs: Any,
) -> BenchmarkOperatorResult:
    """
    Run a list of benchmarks with a given set of inputs and kwargs.
    Kwargs in this case are the command-line arguments available in tritonbench.

    The single_input is a list of arguments that will be passed to the benchmark function all together

    Example:

    def triton_add(x, y):
        ...

    benchmarks = {
        "triton_add": triton_add,
        "triton_add2": triton_add,
    }
    size = 2**12
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    dynamic_run_once(benchmarks=benchmarks, single_input=[x, y], benchmark_name="vector_add")
    """

    def input_iterator(*args):
        def generator():
            yield args

        return generator

    input_generator = input_iterator(*single_input)
    output = dynamic_run(benchmarks, input_generator, silent=silent, **kwargs)
    return output
