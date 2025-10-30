"""
Input loader for Grouped GEMM operator.
"""

from typing import Any, Callable

from tritonbench.operator_loader.aten.input_loader import OperatorInputLoader


class InputLoader(OperatorInputLoader):
    def __init__(self, tritonbench_op: str, input_config: Any):
        super().__init__(tritonbench_op.name, input_config)
        self.op = tritonbench_op

    def get_input_iter(
        self,
    ) -> Callable:
        shapes = [eval(inp)[1] for inp, _cnt in self.operator_db[self.op_name].items()]
        parsed = []
        for entry in shapes:
            B_shape = (
                eval(entry["B"]) if isinstance(entry["B"], str) else tuple(entry["B"])
            )
            A_shapes = [
                eval(a) if isinstance(a, str) else tuple(a) for a in entry["A_list"]
            ]
            parsed.append((A_shapes, B_shape))

        # Set shapes on the operator
        self.op.external_shapes = parsed
        return self.op.get_input_iter
