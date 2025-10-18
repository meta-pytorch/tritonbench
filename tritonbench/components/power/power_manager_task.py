from tritonbench.components.tasks.manager import ManagerTask


class PowerManagerTask(ManagerTask):
    def __init__(self,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(timeout, extra_env)
    
    def start_task(self) -> None:
        self.make_instance(
            "power_manager",
            "tritonbench.components.power.power_manager",
            "PowerManager",
            "pm"
        )

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def start_monitor(self):
        pm = globals()["pm"]
        pm.start()

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def stop_monitor(self):
        pm = globals()["pm"]
        pm.stop()
