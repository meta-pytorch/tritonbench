"""
Input loader for TritonBench flex_attention shape files (--input-loader).

Each JSON entry's ``inputs`` is the stringified ``"((), {kwargs})"`` convention
used by the other ops, where kwargs carry the flex_attention shape and mask type:
``{'B', 'Hq', 'M', 'Hkv', 'N', 'D', 'mod_type'}`` (all string-valued). The tuple
built for each entry is derived through the operator's shared ``_build_input`` so
it matches the hardcoded ``get_input_iter`` path exactly.
"""

import ast
import logging
import random
from typing import Any, Callable

import torch
from tritonbench.operator_loader.aten.input_loader import OperatorInputLoader

logger = logging.getLogger(__name__)


class InputLoader(OperatorInputLoader):
    def __init__(self, tritonbench_op: str, input_config: Any):
        super().__init__(tritonbench_op.name, input_config)
        self.op = tritonbench_op

    def get_input_iter(self) -> Callable:
        # inputs use the "((args), {kwargs})" convention; parse the kwargs dict
        # with a safe literal parser (never eval) since --input-loader configs
        # come from external JSON files.
        specs = [
            ast.literal_eval(inp)[1]
            for inp, _cnt in self.operator_db[self.op_name].items()
        ]

        def _inner():
            random.seed(42)
            torch.manual_seed(42)
            for spec in specs:
                B = int(spec["B"])
                Hq = int(spec["Hq"])
                M = int(spec["M"])
                Hkv = int(spec["Hkv"])
                N = int(spec["N"])
                D = int(spec["D"])
                mod_type = spec["mod_type"]
                q_shape = (B, Hq, M, D)
                kv_shape = (B, Hkv, N, D)
                yield self.op._build_input(q_shape, kv_shape, mod_type)

        return _inner
