# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# it is disabled by default pending compiler fixes and proper runtime gates.
@triton.jit
def _nvfp4_gemm_ws_swizzled(  # noqa: TR001
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    rep_m: tl.constexpr,
    rep_n: tl.constexpr,
    rep_k: tl.constexpr,
    out_dtype: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _k in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=True):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = (
            scale_a.reshape(rep_m, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        )
        scale_b = (
            scale_b.reshape(rep_n, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_N, BLOCK_K // VEC_SIZE)
        )

        accumulator = tl.dot_scaled(
            a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator
        )

        offs_k_a += BLOCK_K // 2
        offs_k_b += BLOCK_K // 2
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(out_dtype))


def _triton_out_dtype(tensor: torch.Tensor):
    return getattr(tl, str(tensor.dtype).split(".")[1])


def triton_autows_nvfp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    m: int,
    n: int,
    k: int,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
    vec_size: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """AutoWS NVFP4 block-scaled GEMM for TritonBench nvfp4_gemm inputs.

    This backend currently does not work: the Meta AutoWS compiler path crashes
    in NVGPUWarpSpecialization while lowering the block-scaled MMA scale operand
    into TMEM.
    """
    if not hasattr(triton, "knobs"):
        raise NotImplementedError("triton_autows_nvfp4_gemm requires Meta Triton")
    if m % block_m != 0 or n % block_n != 0 or k % block_k != 0:
        raise NotImplementedError(
            "triton_autows_nvfp4_gemm requires M/N/K multiples of "
            f"{block_m}/{block_n}/{block_k}"
        )
    if block_k % (vec_size * 4) != 0:
        raise NotImplementedError(
            "triton_autows_nvfp4_gemm requires BLOCK_K % (VEC_SIZE * 4) == 0"
        )

    a_packed = a.contiguous().view(torch.uint8)
    b_packed = b.T.contiguous().view(torch.uint8)
    c = torch.empty((m, n), device=a.device, dtype=out_dtype)

    rep_m = block_m // 128
    rep_n = block_n // 128
    rep_k = block_k // vec_size // 4
    scale_k_blocks = k // vec_size // 4
    scale_a_5d = scale_a.contiguous().reshape(1, m // 128, scale_k_blocks, 2, 256)
    scale_b_5d = scale_b.contiguous().reshape(1, n // 128, scale_k_blocks, 2, 256)

    desc_a = TensorDescriptor.from_tensor(a_packed, [block_m, block_k // 2])
    desc_b = TensorDescriptor.from_tensor(b_packed, [block_n, block_k // 2])
    desc_c = TensorDescriptor.from_tensor(c, [block_m, block_n])
    a_scale_desc = TensorDescriptor.from_tensor(
        scale_a_5d, block_shape=[1, rep_m, rep_k, 2, 256]
    )
    b_scale_desc = TensorDescriptor.from_tensor(
        scale_b_5d, block_shape=[1, rep_n, rep_k, 2, 256]
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=a.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), 1)

    assert triton.knobs.nvidia.use_meta_ws, (
        "triton_autows_nvfp4_gemm requires TRITON_USE_META_WS=1"
    )
    assert triton.knobs.nvidia.disable_wsbarrier_reorder, (
        "triton_autows_nvfp4_gemm currently requires TRITON_DISABLE_WSBARRIER_REORDER=1"
    )
    # TODO: Tune max registers across partitions.
    _nvfp4_gemm_ws_swizzled[grid](
        desc_a,
        a_scale_desc,
        desc_b,
        b_scale_desc,
        desc_c,
        m,
        n,
        k,
        vec_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        rep_m=rep_m,
        rep_n=rep_n,
        rep_k=rep_k,
        out_dtype=_triton_out_dtype(c),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# Triton TR001: this coverage backend intentionally keeps a fixed config; it is
# disabled by default because it currently does not compile through Meta AutoWS.
@triton.jit
def _nvfp4_gemm_persistent_ws(  # noqa: TR001
    a_desc,  # FP4 e2m1 packed [M, K//2]
    a_scale_desc,  # 5D swizzled scale-A descriptor [1, rep_m, rep_k, 2, 256] u8
    b_desc,  # FP4 e2m1 packed [N, K//2]
    b_scale_desc,  # 5D swizzled scale-B descriptor [1, rep_n, rep_k, 2, 256] u8
    c_desc,  # output [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    rep_m: tl.constexpr,
    rep_n: tl.constexpr,
    rep_k: tl.constexpr,
    out_dtype: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_K)

    # Persistent tile loop -- the loop Meta AutoWS specializes. Each iteration is
    # an independent output tile; the load(A/B/scaleA/scaleB) -> scaled-MMA ->
    # store pipeline spans iterations.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        pid_m = tile_id % num_pid_m
        pid_n = tile_id // num_pid_m
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        offs_scale_m = pid_m * rep_m
        offs_scale_n = pid_n * rep_n

        offs_k_a = 0
        offs_scale_k = 0
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # Inner K-reduction loop uses plain range() (constexpr trip count).
        for _ in range(k_tiles):
            a = a_desc.load([offs_am, offs_k_a])
            b = b_desc.load([offs_bn, offs_k_a])
            scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
            scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

            # Unswizzle the hardware-packed scale layout into the rank-2
            # [BLOCK, BLOCK_K//VEC] layout tl.dot_scaled expects.
            scale_a = (
                scale_a.reshape(rep_m, rep_k, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
            )
            scale_b = (
                scale_b.reshape(rep_n, rep_k, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_N, BLOCK_K // VEC_SIZE)
            )

            accumulator = tl.dot_scaled(
                a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator
            )

            offs_k_a += BLOCK_K // 2  # 2 fp4 elems / byte along K
            offs_scale_k += rep_k

        c_desc.store([offs_am, offs_bn], accumulator.to(out_dtype))


def triton_autows_nvfp4_gemm_persistent(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    m: int,
    n: int,
    k: int,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
    vec_size: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """PERSISTENT AutoWS NVFP4 block-scaled GEMM (K-5).

    Same NVFP4 block-scaled MMA math and swizzled-5D scale layout as
    ``triton_autows_nvfp4_gemm`` but expressed as a persistent kernel (``grid =
    #SMs`` with an outer warp-specialized tile loop) so Meta AutoWS has a loop to
    specialize.

    NOTE: this candidate currently does NOT compile through Meta AutoWS -- the
    ``NVGPUWarpSpecialization`` pass crashes staging the rank-2 ``dot_scaled``
    scale operand into TMEM via ``tmem_copy`` (block-scaled-MMA WS-crash class;
    the persistent structure does not avoid it). The source math is correct
    (matches a dequantized reference with ``warp_specialize=False``); the backend
    is carried disabled for coverage.
    """
    if not hasattr(triton, "knobs"):
        raise NotImplementedError(
            "triton_autows_nvfp4_gemm_persistent requires Meta Triton"
        )
    if m % block_m != 0 or n % block_n != 0 or k % block_k != 0:
        raise NotImplementedError(
            "triton_autows_nvfp4_gemm_persistent requires M/N/K multiples of "
            f"{block_m}/{block_n}/{block_k}"
        )
    if block_k % (vec_size * 4) != 0:
        raise NotImplementedError(
            "triton_autows_nvfp4_gemm_persistent requires BLOCK_K % (VEC_SIZE * 4) == 0"
        )

    a_packed = a.contiguous().view(torch.uint8)
    b_packed = b.T.contiguous().view(torch.uint8)
    c = torch.empty((m, n), device=a.device, dtype=out_dtype)

    rep_m = block_m // 128
    rep_n = block_n // 128
    rep_k = block_k // vec_size // 4
    scale_k_blocks = k // vec_size // 4
    scale_a_5d = scale_a.contiguous().reshape(1, m // 128, scale_k_blocks, 2, 256)
    scale_b_5d = scale_b.contiguous().reshape(1, n // 128, scale_k_blocks, 2, 256)

    desc_a = TensorDescriptor.from_tensor(a_packed, [block_m, block_k // 2])
    desc_b = TensorDescriptor.from_tensor(b_packed, [block_n, block_k // 2])
    desc_c = TensorDescriptor.from_tensor(c, [block_m, block_n])
    a_scale_desc = TensorDescriptor.from_tensor(
        scale_a_5d, block_shape=[1, rep_m, rep_k, 2, 256]
    )
    b_scale_desc = TensorDescriptor.from_tensor(
        scale_b_5d, block_shape=[1, rep_n, rep_k, 2, 256]
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=a.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    grid = (min(num_sms, num_tiles),)

    assert triton.knobs.nvidia.use_meta_ws, (
        "triton_autows_nvfp4_gemm_persistent requires TRITON_USE_META_WS=1"
    )
    # TODO: Tune max registers across partitions.
    _nvfp4_gemm_persistent_ws[grid](
        desc_a,
        a_scale_desc,
        desc_b,
        b_scale_desc,
        desc_c,
        m,
        n,
        k,
        NUM_SMS=num_sms,
        VEC_SIZE=vec_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        rep_m=rep_m,
        rep_n=rep_n,
        rep_k=rep_k,
        out_dtype=_triton_out_dtype(c),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
