import argparse
import os
import types
from typing import Dict, Generator, List, Optional

import torch
import yaml
from tritonbench.utils.triton_op import BenchmarkOperator, register_benchmark

# The config file defines available ATen operators and their corresponding input shapes.
ATEN_CONFIG_YAML = os.path.join(os.path.dirname(__file__), "config.yaml")
aten = torch.ops.aten


class AtenOperator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.aten_op = eval(self.aten_op_name)
        if not self.tb_args.input_loader:
            self.tb_args.input_loader = self.aten_op_input

    def get_input_iter(self) -> Generator:
        raise NotImplementedError("get_input_iter is not implemented for AtenOperator.")

    def eager(self, *input):
        args, kwargs = input
        eager = lambda: self.aten_op(*args, **kwargs)
        return lambda: eager()

    def inductor(self, *input):
        args, kwargs = input
        eager = lambda: self.aten_op(*args, **kwargs)
        compiled_fn = torch.compile(eager)
        return lambda: compiled_fn()


def get_default_aten_op_input(aten_op_name: str) -> str:
    """
    Return the default input for the given aten op name.
    """
    with open(ATEN_CONFIG_YAML, "r") as f:
        config = yaml.safe_load(f)
    return config[aten_op_name]


def get_aten_loader_cls_by_name(aten_op_name: str, aten_op_input: Optional[str] = None):
    """
    Return a class generated from the given aten op name and input.
    If input is not provided, use the default input from the config file.
    """
    op_cls_name = aten_op_name.replace(".", "_")
    module_name = f"tritonbench.operator_loader.aten.{op_cls_name}"
    op_name_module = types.ModuleType(module_name)
    op_class = AtenOperator
    op_class.__module__ = module_name
    op_name_module.Operator = op_class
    op_class.name = op_cls_name
    op_class.aten_op_name = aten_op_name
    op_class.aten_op_input = (
        aten_op_input if aten_op_input else get_default_aten_op_input(aten_op_name)
    )
    # register two backends for each aten op: eager and inductor
    register_benchmark(operator_name=op_cls_name, func_name="eager", baseline=True)(
        op_class.eager
    )
    register_benchmark(operator_name=op_cls_name, func_name="inductor", baseline=False)(
        op_class.inductor
    )
    return op_class


def list_aten_ops() -> Dict[str, str]:
    """
    Load all ATen operators from the config file.
    """
    with open(ATEN_CONFIG_YAML, "r") as f:
        config = yaml.safe_load(f)
    return config
