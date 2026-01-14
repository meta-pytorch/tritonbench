import gc
import threading
from typing import Any, Dict, Optional

from tritonbench.components.tasks.base import run_in_worker, TaskBase
from tritonbench.components.workers import subprocess_worker


class ManagerTask(TaskBase):
    # The ManagerTask may (and often does) consume significant system resources.
    # In order to ensure that runs do not interfere with each other, we only
    # allow a single ManagerTask to exist at a time.
    _lock = threading.Lock()

    def __init__(
        self,
        obj_name: str,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        gc.collect()  # Make sure previous task has a chance to release the lock
        assert self._lock.acquire(blocking=False), "Failed to acquire lock."

        self._obj_name = obj_name
        self._worker = subprocess_worker.SubprocessWorker(
            timeout=timeout, extra_env=extra_env
        )

    @run_in_worker(scoped=True)
    @staticmethod
    def make_instance(
        module_path: str,
        package: Optional[str],
        class_name: str,
    ) -> None:
        import importlib
        import os
        import traceback

        # required as this is in child process
        from tritonbench.components.power.power_manager import PowerManager

        module = importlib.import_module(module_path, package=package)
        Ctor = getattr(module, class_name)

        # Populate global namespace so subsequent calls to worker.run can access `Model`
        globals()["Ctor"] = Ctor
        globals()["manager"] = Ctor()

    # Set attribute from the object in the worker process
    @run_in_worker(scoped=True)
    @staticmethod
    def set_manager_attribute(
        attr: str, value: Any, field: str = None, classattr: bool = False
    ) -> None:
        if classattr:
            manager = globals()["manager"].__class__
        else:
            manager = globals()["manager"]
        if hasattr(manager, attr):
            if field:
                manager_attr = getattr(manager, attr)
                setattr(manager_attr, field, value)
            else:
                setattr(manager, attr, value)

    # Get attribute from the object in the worker process
    @run_in_worker(scoped=True)
    @staticmethod
    def get_manager_attribute(
        attr: str, field: str = None, classattr: bool = False
    ) -> Any:
        if classattr:
            manager = globals()["manager"].__class__
        else:
            manager = globals()["manager"]
        if hasattr(manager, attr):
            if field:
                manager_attr = getattr(manager, attr)
                return getattr(manager_attr, field)
            else:
                return getattr(manager, attr)
        else:
            return None

    def gc_collect(self) -> None:
        self.worker.run(
            """
            import gc
            gc.collect()
        """
        )

    def del_task(self) -> None:
        self.worker.run(
            f"""
            del {self._obj_name}
        """
        )
        self.gc_collect()

    def __del__(self) -> None:
        self._lock.release()

    @property
    def worker(self) -> subprocess_worker.SubprocessWorker:
        return self._worker
