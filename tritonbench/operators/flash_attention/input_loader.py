import ast

import torch
from tritonbench.operator_loader.aten.input_loader import OperatorInputLoader


class InputLoader(OperatorInputLoader):
    def __init__(self, op, input_config):
        super().__init__(op.name, input_config)
        self.op = op

    def get_input_iter(self):
        specs = [ast.literal_eval(inp)[1] for inp in self.operator_db[self.op.name]]

        def inputs():
            for spec in specs:
                dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[spec["dtype"]]
                shape = tuple(int(spec[k]) for k in ("Z", "H", "N_CTX", "HEAD_DIM"))
                self.op.causal = spec["causal"] == "True"
                self.op.sm_scale = shape[-1] ** -0.5
                yield tuple(
                    torch.randn(
                        shape,
                        dtype=dtype,
                        device=self.op.device,
                        requires_grad=self.op.requires_grad,
                    )
                    for _ in range(3)
                )

        return inputs
