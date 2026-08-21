from typing import Callable, Generator, Tuple

import torch
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)

from .kernels import moe_fused_mul_sum


def torch_moe_fused_mul_sum(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor | None,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    expert_outputs = inputs.float()
    if expert_map is not None:
        assert topk_ids is not None
        valid_experts = expert_map[topk_ids] >= 0
        expert_outputs = expert_outputs * valid_experts.unsqueeze(-1)
    return (
        (expert_outputs * topk_weights.float().unsqueeze(-1))
        .sum(dim=1)
        .to(inputs.dtype)
    )


class Operator(BenchmarkOperator):
    """Benchmark vLLM's fused weighted reduction of MoE expert outputs."""

    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]
    DEFAULT_PRECISION = "bf16"
    FWD_ONLY = True

    @register_benchmark(baseline=True)
    def eager(
        self,
        inputs: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor | None,
        expert_map: torch.Tensor | None,
    ) -> Callable:
        return lambda: torch_moe_fused_mul_sum(
            inputs, topk_weights, topk_ids, expert_map
        )

    @register_benchmark(tags=["triton", "vllm"])
    def triton(
        self,
        inputs: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor | None,
        expert_map: torch.Tensor | None,
    ) -> Callable:
        return lambda: moe_fused_mul_sum(inputs, topk_weights, topk_ids, expert_map)

    @register_x_val(label="(tokens, top_k, hidden, expert_map)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int, bool]:
        inputs, _, _, expert_map = example_inputs
        return (*inputs.shape, expert_map is not None)

    def get_input_iter(self) -> Generator:
        shapes = [
            (1, 2, 4096),
            (4, 8, 4096),
            (16, 8, 4096),
            (64, 8, 4096),
            (256, 8, 4096),
            (1024, 8, 4096),
        ]
        num_experts = 64
        for num_tokens, top_k, hidden_size in shapes:
            inputs = torch.randn(
                num_tokens,
                top_k,
                hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
            topk_weights = torch.softmax(
                torch.randn(
                    num_tokens,
                    top_k,
                    device=self.device,
                    dtype=torch.float32,
                ),
                dim=-1,
            ).to(self.dtype)

            yield inputs, topk_weights, None, None

            topk_ids = torch.randint(
                0,
                num_experts,
                (num_tokens, top_k),
                device=self.device,
                dtype=torch.int32,
            )
            expert_map = torch.arange(
                num_experts, device=self.device, dtype=torch.int32
            )
            expert_map[::4] = -1
            yield inputs, topk_weights, topk_ids, expert_map
