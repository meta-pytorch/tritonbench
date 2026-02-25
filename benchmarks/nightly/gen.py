# Generate the TRITONBENCH_CONFIG autogen.yaml for nightly benchmark
import os
from pathlib import Path
from typing import Any, Dict

from ..common import setup_tritonbench_cwd, REPO_PATH

setup_tritonbench_cwd()

import yaml

CURRENT_PATH = Path(os.path.abspath(__file__)).parent
OUTPUT_PATH = CURRENT_PATH.joinpath("autogen.yaml")


def get_metadata(name: str, path: Path = CURRENT_PATH) -> Any:
    fpath = os.path.join(path, f"{name}.yaml")
    with open(fpath, "r") as f:
        return yaml.safe_load(f)


NIGHTLY_RUN_CONFIG = get_metadata("run_config", path=CURRENT_PATH)


def gen_run_from_config(config: Dict[str, Any]=NIGHTLY_RUN_CONFIG) -> Dict[str, Any]:
    tritonbench_run_configs = get_benchmark_config_with_tags(
        tags=config["run_config"]["tags"],
        with_backwards=config["run_config"]["with_backwards"],
        metrics=config["run_config"]["metrics"],
    )
    # add manual benchmarks
    disabled = config.get("disabled", [])
    overridden = config.get("overridden", {})
    for benchmark in disabled:
        if benchmark in run_configs:
            tritonbench_run_configs[benchmark]["disabled"] = True
    for benchmark in overridden:
        if not benchmark in overridden:
            tritonbench_run_configs[benchmark] = overridden[benchmark].copy()
            continue
        tritonbench_run_configs[benchmark]["args"] = overridden[benchmark]["args"]
    return tritonbench_run_configs


def run():
    runs = gen_run_from_config()
    with open(OUTPUT_PATH, "w") as f:
        yaml.safe_dump(runs, f, sort_keys=False)


if __name__ == "__main__":
    run()
