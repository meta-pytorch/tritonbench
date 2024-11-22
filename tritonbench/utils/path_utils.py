import os
import sys

from pathlib import Path
from typing import List

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
SUBMODULE_PATH = REPO_PATH.joinpath("submodules")


class add_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class add_ld_library_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.os_environ = os.environ.copy()
        library_path = os.environ.get("LD_LIBRARY_PATH")
        if not library_path:
            os.environ["LD_LIBRARY_PATH"] = self.path
        else:
            os.environ["LD_LIBRARY_PATH"] = f"{library_path}:{self.path}"

    def __exit__(self, exc_type, exc_value, traceback):
        os.environ = self.os_environ.copy()


def _find_param_loc(params, key: str) -> int:
    try:
        return params.index(key)
    except ValueError:
        return -1


def _remove_params(params, loc):
    if loc == -1:
        return params
    if loc == len(params) - 1:
        return params[:loc]
    if params[loc + 1].startswith("--"):
        return params[:loc] + params[loc + 1 :]
    if loc == len(params) - 2:
        return params[:loc]
    return params[:loc] + params[loc + 2 :]


def add_cmd_parameter(args: List[str], name: str, value: str) -> List[str]:
    args.append(name)
    args.append(value)
    return args


def remove_cmd_parameter(args: List[str], name: str) -> List[str]:
    loc = _find_param_loc(args, name)
    return _remove_params(args, loc)
