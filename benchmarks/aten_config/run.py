import argparse
import json
import os

import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATEN_RUN_CONFIGS_DIR = os.path.join(REPO_DIR, "tritonbench/data/input_configs")
OSS_MODEL_SUITES = [
    "torchbench_train",
    "hf_train",
    "timm_train",
]
ATEN_MAPPING = {}


def get_aten_ops(obj):
    return [key for key in obj.keys() if key.startswith("aten")]


def reverse_dict(data):
    reversed_data = {}
    for key, value in data.items():
        suite, model = key
        for op in value:
            reversed_data[op] = f"{suite}/{model}.json"
    return reversed_data


def get_jsons(dir):
    return [json_file for json_file in os.listdir(dir) if json_file.endswith(".json")]


def run(output):
    for suite in OSS_MODEL_SUITES:
        for json_file in get_jsons(os.path.join(ATEN_RUN_CONFIGS_DIR, suite)):
            with open(os.path.join(ATEN_RUN_CONFIGS_DIR, suite, json_file)) as f:
                model = os.path.basename(json_file).split(".")[0]
                ATEN_MAPPING[(suite, model)] = []
                print("Reading model: ", model, " for suite: ", suite)
                data = json.load(f)
                for op in get_aten_ops(data):
                    ATEN_MAPPING[(suite, model)].append(op)
    reversed_mapping = reverse_dict(ATEN_MAPPING)
    with open(output, "w") as fp:
        yaml.safe_dump(reversed_mapping, fp, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="output file path")
    args = parser.parse_args()
    run(args.output)
