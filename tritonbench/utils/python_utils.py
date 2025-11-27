import inspect
from contextlib import contextmanager


@contextmanager
def try_import(cond_name: str):
    frame = inspect.currentframe().f_back.f_back
    _caller_globals = frame.f_globals
    try:
        yield
        _caller_globals[cond_name] = True
    except (ImportError, ModuleNotFoundError, OSError) as e:
        _caller_globals[cond_name] = False
