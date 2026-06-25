import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _set_tma_allocator(device: torch.device) -> None:
    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)


def _set_block_size_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    epilogue_subtile = nargs["EPILOGUE_SUBTILE"]

    nargs["a_desc"].block_shape = [block_m, block_k]
    nargs["b_desc"].block_shape = [block_n, block_k]
    nargs["out_desc"].block_shape = [block_m, block_n // epilogue_subtile]


def _get_autotune_configs():
    configs = []
    for block_m in [64, 128, 256]:
        for block_n in [64, 128, 256]:
            for block_k in [64, 128, 256]:
                for num_warps in [4, 8]:
                    for num_stages in range(2, 8):
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "EPILOGUE_SUBTILE": 1,
                                    "TWO_CTAS": True,
                                    "DATA_PARTITION_FACTOR": 1,
                                    "GROUP_SIZE": 8,
                                    "GROUP_BY_N": False,
                                },
                                num_warps=num_warps,
                                num_stages=num_stages,
                                pre_hook=_set_block_size_hook,
                                ctas_per_cga=(2, 1, 1),
                                early_tma_store_lowering=True,
                            )
                        )
    return configs


def _prune_configs(configs, named_args, **kwargs):
    m = named_args["M"]
    n = named_args["N"]
    k = named_args["K"]
    pruned = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        if m % block_m != 0 or n % block_n != 0 or k % block_k != 0:
            continue
        pruned.append(config)
    return pruned


@triton.jit
def _compute_pid(
    tile_id,
    num_pid_in_group,
    grid_m,
    grid_n,
    GROUP_SIZE: tl.constexpr,
    GROUP_BY_N: tl.constexpr,
):
    if GROUP_BY_N:
        group_id = tile_id // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE
        group_n = tl.minimum(grid_n - first_pid_n, GROUP_SIZE)
        pair_id = (tile_id % num_pid_in_group) // 2
        pair_lane = tile_id % 2
        pid_n = first_pid_n + (pair_id % group_n)
        pid_m = pair_lane + 2 * (pair_id // group_n)
    else:
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE
        group_m = tl.minimum(grid_m - first_pid_m, GROUP_SIZE)
        pid_m = first_pid_m + (tile_id % group_m)
        pid_n = (tile_id % num_pid_in_group) // group_m
    return pid_m, pid_n


@triton.jit
def _subtile_accumulator(
    acc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_FACTOR: tl.constexpr,
):
    if SUBTILE_FACTOR == 1:
        return (acc,)
    else:
        tl.static_assert(BLOCK_N % 2 == 0)
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        left, right = tl.split(acc)
        left_subtiles = _subtile_accumulator(
            left,
            BLOCK_M,
            BLOCK_N // 2,
            SUBTILE_FACTOR // 2,
        )
        right_subtiles = _subtile_accumulator(
            right,
            BLOCK_M,
            BLOCK_N // 2,
            SUBTILE_FACTOR // 2,
        )
        return left_subtiles + right_subtiles


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _gemm_rmsnorm_kernel(
    a_ptr,
    a_desc,
    b_desc,
    out_desc,
    M,
    N,
    K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    TWO_CTAS: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_BY_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    if TWO_CTAS:
        grid_m = (grid_m + 1) // 2 * 2
    grid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = grid_m * grid_n

    if GROUP_BY_N:
        num_pid_in_group = GROUP_SIZE * grid_m
    else:
        num_pid_in_group = GROUP_SIZE * grid_n

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        warp_specialize=True,
        data_partition_factor=DATA_PARTITION_FACTOR,
        separate_epilogue_store=True,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id,
            num_pid_in_group,
            grid_m,
            grid_n,
            GROUP_SIZE,
            GROUP_BY_N,
        )
        offs_am = (pid_m * BLOCK_M).to(tl.int32)
        offs_bn = (pid_n * BLOCK_N).to(tl.int32)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        row_sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for ki in range(0, k_tiles):
            offs_k = (ki * BLOCK_K).to(tl.int32)
            rows = offs_am + tl.arange(0, BLOCK_M)
            cols_k = offs_k + tl.arange(0, BLOCK_K)
            a = a_desc.load([offs_am, offs_k])
            a_rms = tl.load(
                a_ptr + rows[:, None] * stride_am + cols_k[None, :] * stride_ak,
                mask=(rows[:, None] < M) & (cols_k[None, :] < K),
                other=0.0,
            )
            b = b_desc.load([offs_bn, offs_k])
            a_f32 = a_rms.to(tl.float32)
            row_sumsq += tl.sum(a_f32 * a_f32, axis=1)
            accumulator += tl.dot(
                a,
                b.T,
                allow_tf32=False,
                two_ctas=TWO_CTAS,
            )

        rstd = tl.rsqrt(row_sumsq / K + EPS)
        accumulator *= rstd[:, None]

        subtiles = _subtile_accumulator(
            accumulator,
            BLOCK_M,
            BLOCK_N,
            EPILOGUE_SUBTILE,
        )
        for i in tl.static_range(EPILOGUE_SUBTILE):
            offs_cn_i = offs_bn + i * (BLOCK_N // EPILOGUE_SUBTILE)
            out_desc.store([offs_am, offs_cn_i], subtiles[i].to(tl.bfloat16))


def triton_autows_fused_rmsnorm_gemm(
    x: torch.Tensor,
    b_weighted_t: torch.Tensor,
    eps: float,
    *,
    num_sms: int = 148,
) -> torch.Tensor:
    m, k = x.shape
    n, wk = b_weighted_t.shape
    assert wk == k

    _set_tma_allocator(x.device)

    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    a_desc = TensorDescriptor(x, [m, k], x.stride(), [1, 1])
    b_desc = TensorDescriptor(
        b_weighted_t,
        [n, k],
        b_weighted_t.stride(),
        [1, 1],
    )
    out_desc = TensorDescriptor(out, [m, n], out.stride(), [1, 1])

    _gemm_rmsnorm_kernel[(num_sms, 1, 1)](
        x,
        a_desc,
        b_desc,
        out_desc,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        eps,
        NUM_SMS=num_sms,
    )
    return out
