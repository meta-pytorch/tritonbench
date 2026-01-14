import dataclasses
import gc
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tritonbench.components.tasks import base as base_task
from tritonbench.components.workers import subprocess_worker


class Worker(subprocess_worker.SubprocessWorker):
    """Run subprocess using taskset if CPU affinity is set.

    When GOMP_CPU_AFFINITY is set, importing `torch` in the main process has
    the very surprising effect of changing the threading behavior in the
    subprocess. (See https://github.com/pytorch/pytorch/issues/49971 for
    details.) This is a problem, because it means that the worker is not
    hermetic and also tends to force the subprocess torch to run in single
    threaded mode which drastically skews results.

    This can be ameliorated by calling the subprocess using `taskset`, which
    allows the subprocess PyTorch to properly bind threads.
    """

    @property
    def args(self) -> List[str]:
        affinity = os.environ.get("GOMP_CPU_AFFINITY", "")
        return (["taskset", "--cpu-list", affinity] if affinity else []) + super().args


@dataclasses.dataclass(frozen=True)
class OpDetails:
    """Static description of what a particular TritonBench operator supports.

    When parameterizing tests, we only want to generate sensible ones.
    (e.g. Those where an operator can be imported and supports the feature to be
    tested or benchmarked.) This requires us to import the operator; however many
    of the operators are EXTREMELY stateful, and even importing them consumes
    significant system resources. As a result, we only want one (or a few)
    alive at any given time.

    Note that affinity cannot be solved by simply calling `torch.set_num_threads`
    in the child process; this will cause PyTorch to use all of the cores but
    at a much lower efficiency.

    This class describes what a particular operator does and does not support, so
    that we can release the underlying subprocess but retain any pertinent
    metadata.
    """

    name: str
    exists: bool


class OpTask(base_task.TaskBase):
    # The worker may (and often does) consume significant system resources.
    # In order to ensure that runs do not interfere with each other, we only
    # allow a single OpTask to exist at a time.
    _lock = threading.Lock()

    def __init__(
        self,
        name: str,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
        save_output_dir: Optional[Path] = None,
    ) -> None:
        gc.collect()  # Make sure previous task has a chance to release the lock
        assert self._lock.acquire(blocking=False), "Failed to acquire lock."

        self._op_name = name
        self._worker = Worker(
            timeout=timeout, extra_env=extra_env, save_output_dir=save_output_dir
        )

        self.worker.run("import torch")
        self._details: OpDetails = OpDetails(
            **self._maybe_import_operator(
                package=__name__,
                op_name=name,
            )
        )

    # =========================================================================
    # == Import Operator in the child process ====================================
    # =========================================================================

    @property
    def worker(self) -> subprocess_worker.SubprocessWorker:
        return self._worker

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def _maybe_import_operator(package: str, op_name: str) -> Dict[str, Any]:
        import importlib
        import os
        import traceback

        from tritonbench.operators import load_opbench_by_name

        Operator = load_opbench_by_name(op_name)

        # Populate global namespace so subsequent calls to worker.run can access `Operator`
        globals()["Operator"] = Operator

        # This will be used to populate a `OpDetails` instance in the parent.
        return {
            "name": op_name,
            "exists": Operator is not None,
        }

    # =========================================================================
    # == Instantiate a concrete `op` instance ==============================
    # =========================================================================

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def make_operator_instance(
        args: List[str],
    ) -> None:
        from tritonbench.utils.parser import get_parser

        parser = get_parser()
        tb_args, extra_args = parser.parse_known_args(args)
        Operator = globals()["Operator"]
        parser = get_parser()
        op = Operator(
            tb_args=tb_args,
            extra_args=extra_args,
        )

        import gc

        gc.collect()

        if op.device == "cuda":
            maybe_sync = torch.cuda.synchronize
        else:
            maybe_sync = lambda: None

        globals().update(
            {
                "op": op,
                "maybe_sync": maybe_sync,
            }
        )

    # =========================================================================
    # == Forward calls to `op` from parent to worker =======================
    # =========================================================================
    def run(self) -> None:
        self.worker.run(
            """
            op.run()
            maybe_sync()
        """
        )

    # =========================================================================
    # == Get Operator attribute in the child process =============================
    # =========================================================================
    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def get_attribute(
        attr: str,
        field: Optional[str] = None,
        classattr: bool = False,
        method: bool = False,
    ) -> Any:
        if classattr:
            op = globals()["Operator"]
        else:
            op = globals()["op"]
        if hasattr(op, attr):
            if field:
                op_attr = getattr(getattr(op, attr), field)
            else:
                op_attr = getattr(op, attr)
            return op_attr() if method else op_attr
        else:
            return None

    # =========================================================================
    # == Check output is expected in the child process ========================
    # =========================================================================
    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def check_output() -> None:
        op = globals()["op"]
        from tritonbench.utils.triton_op import (
            find_enabled_benchmarks,
            REGISTERED_BENCHMARKS,
        )

        output = op.output
        output_impls = output.result[0][1].keys()
        ci_enabled_impls = find_enabled_benchmarks(
            op.mode, REGISTERED_BENCHMARKS[output.op_name], op._skip
        )
        # Make sure that all the ci_enabled impls are in the output
        assert set(output_impls) == set(ci_enabled_impls), (
            f"output impls: {output_impls} != ci_enabled impls: {ci_enabled_impls}"
        )

    def del_op_instance(self):
        self.worker.run(
            """
            del op
            del maybe_sync
        """
        )
        self.gc_collect()

    def gc_collect(self) -> None:
        self.worker.run(
            """
            import gc
            gc.collect()
        """
        )

    def __del__(self) -> None:
        self._lock.release()
