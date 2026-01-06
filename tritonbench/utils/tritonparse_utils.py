"""
This module provides utility functions for integrating with TritonParse.
TritonParse is a tool for tracing, visualizing, and analyzing Triton kernels.
For more details, see: https://github.com/meta-pytorch/tritonparse
"""

import importlib.util

from .env_utils import is_fbcode


def tritonparse_init(tritonparse_log_path):
    """
    Initializes TritonParse structured logging.

    This function sets up the logging hook to capture Triton compilation
    and launch events. The logs will be stored in a 'raw_logs' subdirectory
    within the specified path. For more details, see:
    https://github.com/pytorch-labs/tritonparse

    Args:
        tritonparse_log_path (str or None): The path to the directory where
            TritonParse logs should be stored. Raw logs will be saved in
            {tritonparse_log_path}/raw_logs/. If None, this function
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
                f"{tritonparse_log_path}/raw_logs",
                enable_trace_launch=True,
                enable_more_tensor_information=True,
            )
            print(
                f"TritonParse structured logging initialized with log path: {tritonparse_log_path}"
            )
        except Exception as e:
            print(f"Warning: Failed to initialize tritonparse: {e}")


def tritonparse_parse(tritonparse_log_path):
    """
    Parses the generated TritonParse logs.

    This function processes the raw logs from the 'raw_logs' subdirectory
    and creates unified, structured trace files in the 'parsed_logs'
    subdirectory. For more details, see:
    https://github.com/pytorch-labs/tritonparse

    Args:
        tritonparse_log_path (str or None): The base path to the directory
            containing TritonParse logs. Raw logs will be read from
            {tritonparse_log_path}/raw_logs/ and parsed logs will be saved to
            {tritonparse_log_path}/parsed_logs/. If None, this function
            does nothing.
    """
    if tritonparse_log_path is not None:
        # capture errors but don't fail the entire script
        try:
            from tritonparse.parse.utils import unified_parse

            if is_fbcode():
                out = None
            else:
                out = f"{tritonparse_log_path}/parsed_logs"
            unified_parse(
                f"{tritonparse_log_path}/raw_logs",
                out=out,
                overwrite=True,
            )
        except Exception as e:
            print(f"Warning: Failed to parse tritonparse log: {e}")
