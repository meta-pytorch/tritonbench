# Generate the TRITONBENCH_CONFIG autogen.yaml for nightly benchmark
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from ..common import setup_tritonbench_cwd

setup_tritonbench_cwd()

import yaml
from tritonbench.metadata.query import get_benchmark_config_with_tags

CURRENT_PATH = Path(os.path.abspath(__file__)).parent
OUTPUT_PATH = CURRENT_PATH.joinpath("autogen.yaml")


def get_metadata(name: str, path: Path = CURRENT_PATH) -> Any:
    fpath = os.path.join(path, f"{name}.yaml")
    with open(fpath, "r") as f:
        return yaml.safe_load(f)


NIGHTLY_RUN_CONFIG = get_metadata("run_config", path=CURRENT_PATH)


def gen_run_from_config(config: Dict[str, Any] = NIGHTLY_RUN_CONFIG) -> Dict[str, Any]:
    tritonbench_run_configs = get_benchmark_config_with_tags(
        tags=config["run_config"]["tags"],
        with_backwards=config["run_config"]["with_backwards"],
        metrics=config["run_config"]["metrics"],
    )
    # add manual benchmarks
    disabled = config.get("disabled", [])
    overrides = config.get("overrides", {})
    for benchmark in disabled:
        if benchmark in tritonbench_run_configs:
            tritonbench_run_configs[benchmark]["disabled"] = True
    for benchmark in overrides:
        tritonbench_run_configs[benchmark] = overrides[benchmark].copy()
    return tritonbench_run_configs


def run():
    runs = gen_run_from_config()
    with open(OUTPUT_PATH, "w") as f:
        yaml.safe_dump(runs, f, sort_keys=False)
    logger.info(f"Generated nightly run config at {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
