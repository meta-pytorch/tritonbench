import importlib

from tritonbench.utils.path_utils import add_path, REPO_PATH


def load_plugin(plugin_name: str):
    module_name, delimiter, func_name = plugin_name.rpartition(".")
    with add_path(REPO_PATH):
        module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func()
