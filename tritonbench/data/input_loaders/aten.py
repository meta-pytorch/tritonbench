"""
Load aten inputs from serialized txt files.
"""

import functools
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Generator

import torch
from torch.testing import make_tensor
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map

logger = logging.getLogger(__name__)


aten = torch.ops.aten
tensor_type = torch._C.TensorType.get()

dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

dtype_abbrs_parsing = {value: key for key, value in dtype_abbrs.items()}

INPUT_CONFIG_DIR = Path(__file__).parent.parent.joinpath("input_configs")


def truncate_inp(arg):
    if arg in dtype_abbrs:
        return dtype_abbrs[arg]
    elif isinstance(arg, torch.device):
        return arg.type
    else:
        return arg


# Serialize Function Call
class FuncCallWrapper:
    def __init__(self, call, *args, **kwargs):
        self.call = call
        self.args = tree_map(truncate_inp, args)
        self.kwargs = tree_map(truncate_inp, kwargs) if kwargs is not None else {}

    def __repr__(self):
        args = ", ".join([repr(arg) for arg in self.args])
        kwargs = "".join(
            [f", {str(key)}={value}" for key, value in self.kwargs.items()]
        )
        out = f"{self.call}({args}{kwargs})".strip('"')
        # f strings introduce quotations we dont want
        for key in dtype_abbrs_parsing:
            out = out.replace(f"'{key}'", key)
        return out


def serialize_sparse_tensor(e):
    if isinstance(e, torch._subclasses.FakeTensor):
        return FuncCallWrapper("ST", list(e.shape), e.dtype, e.layout, e.is_coalesced())
    else:
        return FuncCallWrapper(
            "ST", list(e.shape), e.dtype, e.layout, e.is_coalesced(), e._nnz()
        )


def deserialize_sparse_tensor(size, dtype, layout, is_coalesced, nnz=None):
    raise NotImplementedError("Sparse Tensor generation is not implemented.")


def deserialize_tensor(size, dtype, stride=None):
    if stride is not None:
        out = torch.empty_strided(size, stride, dtype=dtype)
    else:
        out = torch.empty(size, dtype=dtype)
    try:
        out.copy_(make_tensor(size, dtype=dtype, device="cpu"))
    except Exception as e:
        print(e)
        return out
    return out


def contains_tensor(elems):
    for elem in pytree.tree_leaves(elems):
        if isinstance(elem, torch.Tensor):
            return True
    return False


def skip_args(elems):
    for i in pytree.tree_leaves(elems):
        # only shows up in constructors and ops like that
        if isinstance(i, (torch.memory_format, torch.storage.UntypedStorage)):
            return True
    return False


def contains_tensor_types(type):
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )


@functools.lru_cache(None)
def non_compute_operator(op):
    schema = op._schema

    # skip constructors
    if not any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return True
    if "_like" in op.name():
        return True

    # allow in place writes
    if schema.is_mutable:
        return False

    tensor_inps = [arg for arg in schema.arguments if arg.type is tensor_type]
    tensor_outputs = [ret for ret in schema.returns if ret.type is tensor_type]

    # skip aliasing unless there are multiple outputs
    if len(tensor_outputs) != 1:
        return False

    for inp in tensor_inps:
        if inp.alias_info and tensor_outputs[0].alias_info:
            if inp.alias_info.before_set.intersection(
                tensor_outputs[0].alias_info.after_set
            ):
                return True

    return False


def deserialize_args(inps):
    inps = inps.strip().strip("'")
    global_vals = {
        "T": deserialize_tensor,
        "ST": deserialize_sparse_tensor,
        "th": torch,
        "inf": math.inf,
        "torch": torch,
        **dtype_abbrs_parsing,
    }
    # f strings introduce quotations we dont want
    for key in dtype_abbrs_parsing:
        inps = inps.replace(f"'{key}'", key)
    return eval(inps.strip().strip("'").strip('"'), global_vals)


class OperatorInputsLoader:
    def __init__(self, op_name: str, json_file_path: str):
        self.op_name = op_name
        self.operator_db = defaultdict(Counter)

        with open(json_file_path, "r") as f:
            obj = json.load(f)

        for operator in obj:
            op_inps = Counter()
            for inputs in obj[operator]:
                cnt = inputs["count"]
                inps = inputs["inputs"]
                op_inps[inps] += cnt
            self.operator_db[operator] = op_inps
            if "embedding" in str(operator):
                raise RuntimeError(
                    "Embedding inputs not yet implemented, input data cannot be randomized"
                )
        if self.op_name not in self.operator_db:
            raise RuntimeError(f"Could not find {self.op_name} in {json_file_path}.")

    def get_input_iter(
        self,
    ) -> Callable:
        def _input_iter() -> Generator:
            # line[1] represents number of times these inputs occured, ignored for now
            for line in self.operator_db[self.op_name].items():
                inps = line[0]
                args, kwargs = deserialize_args(inps)
                yield (
                    args,
                    kwargs,
                )

        return _input_iter

    def get_all_ops(self):
        for key in self.operator_db.keys():
            try:
                op = eval(key)
            except AttributeError as ae:
                logger.warning("Evaluating an op name into an OpOverload: %s", ae)
                continue
            yield op

    def get_call_frequency(self, op):
        assert (
            str(op) in self.operator_db
        ), f"Could not find {op}, must provide overload"

        count = 0
        for counter in self.operator_db[str(op)].values():
            count += counter
        return count

    def merge(self, other):
        for operator, counter_dict in other.operator_db.items():
            for inps, cnt in counter_dict.items():
                self.operator_db[operator][inps] += cnt


def get_input_iter(tritonbench_op: Any, op: str, input: str) -> Generator:
    aten_op_name = tritonbench_op.aten_op_name
    input_file_path = INPUT_CONFIG_DIR.joinpath(input)
    operator_inputs_loader = OperatorInputsLoader(aten_op_name, input_file_path)
    return operator_inputs_loader.get_input_iter()
