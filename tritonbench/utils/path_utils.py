import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
SUBMODULE_PATH = REPO_PATH.joinpath("submodules")
BUILD_PATH = REPO_PATH.joinpath("build")


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


@contextmanager
def ensure_build_subdir_on_sys_path(subdir: str = ""):
    path = BUILD_PATH.joinpath(subdir)
    path_str = str(path)
    added = False
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
        added = True
    try:
        yield path_str
    finally:
        if added:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def _find_param_loc(params, key: str) -> int:
    try:
        return params.index(key)
    except ValueError:
        return -1


def _param_has_argument(params, index: str) -> bool:
    if index == -1 or index == len(params) - 1:
        return False
    if params[index + 1].startswith("-"):
        return False
    return True


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


def get_cmd_parameter(args: List[str], name: str) -> Optional[Union[str, bool]]:
    loc = _find_param_loc(args, name)
    if loc == -1:
        return None
    if _param_has_argument(args, loc):
        return args[loc + 1]
    return True
