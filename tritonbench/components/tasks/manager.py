from tritonbench.components.tasks.base import TaskBase


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
        self._worker = Worker(timeout=timeout, extra_env=extra_env)

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def make_instance(
        module_path: str, package: Optional[str], class_name: str, obj_name: str
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
        globals()[obj_name] = Ctor()

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
