from typing import Any, Callable, Generator, List, Optional
import torch
import torch.nn as nn 
import argparse
import torch
import triton

from functools import partial

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from .triton_kernel import triton_poi_fused_addmm_relu_view_0

BATCH_SIZE = 960
N_ELEMENTS = 2048

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
        # self.drop1 = nn.Dropout(drop_probs[0])
        # self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        # self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        # self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop1(x)
        # x = self.norm(x)
        # x = self.fc2(x)
        # x = self.drop2(x)
        return x

@torch.no_grad
def run_forward(model, input):
    model.eval()
    # with torch.amp.autocast("cuda", torch.bfloat16):
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
        self.gt_model = Mlp(in_features=512, hidden_features=512 * 4, act_layer=approx_gelu, drop=0, bias=use_bias).cuda().to(torch.bfloat16)
        self.gt_model_copy = Mlp(in_features=512, hidden_features=512 * 4, act_layer=approx_gelu, drop=0, bias=use_bias).cuda().to(torch.bfloat16)
        # self.gt_model = Mlp(in_features=512, hidden_features=512 * 4, act_layer=approx_gelu, drop=0, bias=use_bias).cuda()
        # self.gt_model_copy = Mlp(in_features=512, hidden_features=512 * 4, act_layer=approx_gelu, drop=0, bias=use_bias).cuda()
        self.gt_model_copy.load_state_dict(self.gt_model.state_dict())
        self.compiled_model = torch.compile(self.gt_model_copy, dynamic=False)

    # def get_input_iter(self) -> Generator:
    #     # B, C, T, H, W = 8, 512, 64, 3, 5
    #     # for i in range(10):
    #     #     yield torch.randn((B, T, H, W, C), generator=torch.Generator().manual_seed(i), dtype=torch.bfloat16).cuda()
    #         # yield torch.randn((B, T, H, W, C), generator=torch.Generator().manual_seed(i)).cuda()
    #     # arg0 = torch.randn((2048, 512), dtype=torch.bfloat16).cuda()
    #     # arg1 = torch.randn((2048, ), dtype=torch.bfloat16).cuda()
    #     # arg2 = torch.randn((64, 3, 5, 512), dtype=torch.bfloat16).cuda()
    #     arg0 = torch.randn((512, 2048), dtype=torch.bfloat16).cuda()
    #     arg1 = torch.randn((2048, ), dtype=torch.bfloat16).cuda()
    #     arg2 = torch.randn((960, 512), dtype=torch.bfloat16).cuda()
    #     yield (arg0, arg1, arg2, )

    # @register_benchmark(baseline=True)
    # def gt_mlp(self, input, *args, **kwargs) -> Callable:
    #     return lambda: run_forward(self.gt_model, input)

    # @register_benchmark()
    # def compile_mlp(self, input, *args, **kwargs) -> Callable:
    #    return lambda: run_forward(self.compiled_model, input)

    # @register_benchmark()
    # def cpu(self, arg0, arg1, arg2):
    #     # reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
    #     # arg2_1 = reinterpret_tensor(arg2, (960, 512), (512, 1), 0)
    #     # arg0_1 = reinterpret_tensor(arg0, (512, 2048), (1, 512), 0)
    #     arg1_cpu = arg1.cpu()
    #     arg0_cpu = arg0.cpu()
    #     arg2_cpu = arg2.cpu()
    #     out3 = torch.randn(960, 2048, dtype=torch.bfloat16).cpu()
    #     def _inner():
    #         # out1 = torch.addmm(arg1_cpu, arg2_cpu, arg0_cpu)
    #         # torch.clamp_min(out1, 0, out=out2)
    #         out1 = torch.mm(arg2_cpu, arg0_cpu)
    #         out2 = torch.add(out1, arg1_cpu)
    #         torch.clamp_min(out2, 0, out=out3)
    #         return out2.cuda()
    #     return _inner

    # @register_benchmark()
    # def aten(self, arg0, arg1, arg2):
    #     # reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
    #     # arg2_1 = reinterpret_tensor(arg2, (960, 512), (512, 1), 0)
    #     # arg0_1 = reinterpret_tensor(arg0, (512, 2048), (1, 512), 0)
    #     out3 = torch.randn(960, 2048, dtype=torch.bfloat16).cuda()
    #     def _inner():
    #         out1 = torch.addmm(arg1, arg2, arg0)
    #         # out1 = torch.mm(arg2, arg0)
    #         # out2 = torch.add(out1, arg1)
    #         torch.clamp_min(out1, 0, out=out3)
    #         return out3
    #     return _inner

    # @register_benchmark()
    # def triton(self, arg0, arg1, arg2):
    #     n_elements = 1966080
    #     grid = lambda meta: (triton.cdiv(n_elements, meta['XBLOCK']),)
    #     def _inner():
    #         # stream0 = get_raw_stream(0)
    #         buf0 = torch.mm(arg2, arg0)
    #         # triton_poi_fused_addmm_relu_view_0.run(buf0, arg1, 1966080, stream=stream0)
    #         triton_poi_fused_addmm_relu_view_0[grid](buf0, arg1, 1966080, XBLOCK=1024)
    #         return buf0
    #     return _inner


    def get_input_iter(self) -> Generator:
        # B, C, T, H, W = 8, 512, 64, 3, 5
        # for i in range(10):
        #     yield torch.randn((B, T, H, W, C), generator=torch.Generator().manual_seed(i), dtype=torch.bfloat16).cuda()
            # yield torch.randn((B, T, H, W, C), generator=torch.Generator().manual_seed(i)).cuda()
        arg0 = torch.randn((512, 2048), dtype=torch.bfloat16).cuda()
        arg1 = torch.randn((2048, ), dtype=torch.bfloat16).cuda()
        arg2 = torch.randn((960, 512), dtype=torch.bfloat16).cuda()
        yield (arg0, arg1, arg2)
        return

    @register_benchmark()
    def aten(self, arg0, arg1, arg2):
        out3 = torch.randn(BATCH_SIZE, N_ELEMENTS, dtype=torch.bfloat16).cuda()
        def _inner():
            out2 = torch.addmm(arg1, arg2, arg0)
            # out2 = torch.add(out, arg1)
            torch.clamp_min(out2, 0, out=out3)
            return out3
        return _inner
    
    @register_benchmark()
    def aten2(self, arg0, arg1, arg2):
        out3 = torch.randn(BATCH_SIZE, N_ELEMENTS, dtype=torch.bfloat16).cuda()
        def _inner():
            out = torch.mm(arg2, arg0)
            out2 = torch.add(out, arg1)
            torch.clamp_min(out2, 0, out=out3)
            return out3
        return _inner

    @register_benchmark()
    def triton(self, arg0, arg1, arg2):
        n_elements = BATCH_SIZE * N_ELEMENTS
        grid = lambda meta: (triton.cdiv(n_elements, meta['XBLOCK']),)
        def _inner():
            out = torch.mm(arg2, arg0)
            triton_poi_fused_addmm_relu_view_0[grid](out, arg1, n_elements, XBLOCK=2048)
            return out
        return _inner
