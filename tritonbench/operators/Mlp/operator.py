from typing import Any, Callable, Generator, List, Optional
import torch
import torch.nn as nn 
import argparse
import torch

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

@torch.no_grad
def run_forward(model, input):
    model.eval()
    with torch.amp.autocast("cuda", torch.bfloat16):
        output = model(input)
    return output

def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_bias", action="store_true", help="Whether to enable bias")
    return parser.parse_args(args)

class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        approx_gelu = lambda: nn.ReLU()
        use_bias = parse_op_args(self.extra_args).use_bias
        self.gt_model = Mlp(in_features=512, hidden_features=512 * 4, act_layer=approx_gelu, drop=0, bias=use_bias).cuda()
        self.gt_model_copy = Mlp(in_features=512, hidden_features=512 * 4, act_layer=approx_gelu, drop=0, bias=use_bias).cuda()
        self.gt_model_copy.load_state_dict(self.gt_model.state_dict())
        self.compiled_model = torch.compile(self.gt_model_copy, dynamic=False)
        
    def get_input_iter(self) -> Generator:
        B, C, T, H, W = 8, 512, 64, 3, 5
        for i in range(10):
            yield torch.randn((B, T, H, W, C), generator=torch.Generator().manual_seed(i)).cuda()

    @register_benchmark(baseline=True)
    def gt_mlp(self, input, *args, **kwargs) -> Callable:
        return lambda: run_forward(self.gt_model, input)

    @register_benchmark()
    def compile_mlp(self, input, *args, **kwargs) -> Callable:
        return lambda: run_forward(self.compiled_model, input)
    