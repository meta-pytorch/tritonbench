"""
This module provides utility functions for integrating with TritonParse.
TritonParse is a tool for tracing, visualizing, and analyzing Triton kernels.
For more details, see: https://github.com/meta-pytorch/tritonparse
"""

import importlib.util


def tritonparse_init(tritonparse_log_path):
    """Initializes TritonParse structured logging.

    This function sets up the logging hook to capture Triton compilation
    and launch events. For more details, see:
    https://github.com/meta-pytorch/tritonparse

    Args:
        tritonparse_log_path (str or None): The path to the directory where
            TritonParse logs should be stored. If None, this function
            does nothing.
    """
    if tritonparse_log_path is not None:
        # capture errors but don't fail the entire script
        try:
            if importlib.util.find_spec("tritonparse") is None:
                print(
                    "Warning: tritonparse is not installed. Run 'python install.py --tritonparse' to install it."
                )
                return
            import tritonparse.structured_logging

            tritonparse.structured_logging.init(
                tritonparse_log_path, enable_trace_launch=True
            )
            print(
                f"TritonParse structured logging initialized with log path: {tritonparse_log_path}"
            )
        except Exception as e:
            print(f"Warning: Failed to initialize tritonparse: {e}")


def tritonparse_parse(tritonparse_log_path):
    """Parses the generated TritonParse logs.

    This function processes the raw logs generated during the run and
    creates unified, structured trace files. For more details, see:
    https://github.com/meta-pytorch/tritonparse

    Args:
        tritonparse_log_path (str or None): The path to the directory containing
            the TritonParse logs to be parsed. If None, this function
            does nothing.
    """
    if tritonparse_log_path is not None:
        # capture errors but don't fail the entire script
        try:
            from tritonparse.utils import unified_parse

            unified_parse(tritonparse_log_path)
        except Exception as e:
            print(f"Warning: Failed to parse tritonparse log: {e}")
