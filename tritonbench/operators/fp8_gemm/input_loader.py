"""
Load input shapes from JSON data for FP8 GEMM operator.

Expected op_inputs format:

{
    "fp8_gemm": [
        {
            "count": 1,
            "duration_ms": 0.0106,
            "inputs": "((), {'M': '192', 'N': '512', 'K': '512'})"
        },
        ...
    ]
}
"""

from typing import Callable

from tritonbench.operator_loader.aten.input_loader import OperatorInputLoader


class InputLoader(OperatorInputLoader):
    def __init__(self, tritonbench_op: str, op_name: str, json_file_path: str):
        super().__init__(op_name, json_file_path)
        self.op = tritonbench_op

    def get_input_iter(
        self,
    ) -> Callable:
        shapes = [eval(inp)[1] for inp, _cnt in self.operator_db[self.op_name].items()]
        all_shapes = []
        for entry in shapes:
            M = int(entry["M"])
            N = int(entry["N"])
            K = int(entry["K"])
            all_shapes.append((M, N, K))
        self.op.external_shapes = all_shapes
        return self.op.get_input_iter
