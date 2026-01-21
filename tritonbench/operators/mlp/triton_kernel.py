import triton
import torch


import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

# @triton_heuristics.pointwise(
#     size_hints={'x': 2097152}, 
#     filename=__file__,
#     triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'launch_pdl': False, 'enable_fp_fusion': True},
#     inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '1BA1F624A2AE1B9325912F4B30D7BBD3A5277F921C0D5F915B6AF71FA5CACB3E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 11800576}},
#     min_elem_per_thread=0
# )

@triton.jit
def triton_poi_fused_addmm_relu_view_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2048)
    x2 = xindex
    # tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    # tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.float32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    #tmp4 = triton_helpers.maximum(tmp3, tmp1)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2048, 512), (512, 1))
        assert_size_stride(arg1_1, (2048, ), (1, ))
        assert_size_stride(arg2_1, (64, 3, 5, 512), (7680, 2560, 512, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((960, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(arg2_1, (960, 512), (512, 1), 0), reinterpret_tensor(arg0_1, (512, 2048), (1, 512), 0), out=buf0)
            del arg0_1
            del arg2_1
            buf1 = reinterpret_tensor(buf0, (64, 3, 5, 2048), (30720, 10240, 2048, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.addmm, aten.view, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_view_0.run(buf1, arg1_1, 1966080, stream=stream0)
            del arg1_1
        return (buf1, )
