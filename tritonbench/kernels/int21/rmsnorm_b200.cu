// Copyright 2026 INT21 AI
// SPDX-License-Identifier: MIT

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace {

namespace cg = cooperative_groups;

#define CUDA_KERNEL_CHECK()                                                     \
  do {                                                                         \
    cudaError_t err__ = cudaGetLastError();                                    \
    TORCH_CHECK(err__ == cudaSuccess, "CUDA kernel launch failed: ",           \
                cudaGetErrorString(err__));                                    \
  } while (0)

enum class DType : int {
  kNone = 0,
  kHalf = 1,
  kBFloat16 = 2,
  kFloat = 3,
};

struct Tensor3 {
  void* ptr;
  DType dtype;
  int64_t stride_m;
  int64_t stride_h;
  int64_t stride_n;
};

struct Tensor2 {
  void* ptr;
  DType dtype;
  int64_t stride_m;
  int64_t stride_h;
};

struct Affine {
  void* ptr;
  DType dtype;
  int64_t stride_h;
  int64_t stride_n;
  int per_head;
};

union Float4Raw {
  uint4 raw;
  float elem[4];
};

union U128Raw {
  uint4 u32;
  ulonglong2 u64;
};

template <typename T>
struct Vec128 {
  static constexpr int kElements = 16 / sizeof(T);
  union Storage {
    uint4 raw;
    T elem[kElements];
  };
};

template <typename T>
__device__ __forceinline__ float to_float_t(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float_t<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float_t<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float_t(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half from_float_t<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float_t<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ uint4 ld_global_u128(void const* ptr) {
  uint4 value;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(ptr));
  return value;
}

__device__ __forceinline__ uint4 ld_global_cg_u128(void const* ptr) {
  uint4 value;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(ptr));
  return value;
}

__device__ __forceinline__ uint4 ld_global_ca_u128(void const* ptr) {
  uint4 value;
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(ptr));
  return value;
}

__device__ __forceinline__ uint64_t createpolicy_evict_last_l2(void const* ptr,
                                                               uint32_t size) {
  uint64_t policy;
  asm volatile("createpolicy.range.global.L2::evict_last.b64 %0, [%1], %2, %2;"
               : "=l"(policy)
               : "l"(ptr), "r"(size));
  return policy;
}

__device__ __forceinline__ uint4 ld_global_nc_cache_hint_l2_256_u128(
    void const* ptr,
    uint64_t policy) {
  U128Raw value;
  asm volatile("{\n\t"
               ".reg .b128 value128;\n\t"
               "ld.global.nc.L2::cache_hint.L2::256B.b128 value128, [%2], %3;\n\t"
               "mov.b128 {%0, %1}, value128;\n\t"
               "}"
               : "=l"(value.u64.x), "=l"(value.u64.y)
               : "l"(ptr), "l"(policy)
               : "memory");
  return value.u32;
}

__device__ __forceinline__ void prefetch_global_l2(void const* ptr) {
  asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
}

__device__ __forceinline__ void st_global_u128(void* ptr, uint4 value) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

__device__ __forceinline__ void st_global_cs_u128(void* ptr, uint4 value) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

__device__ __forceinline__ void st_global_v4_f32(float* ptr,
                                                 float x,
                                                 float y,
                                                 float z,
                                                 float w) {
  asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "f"(x), "f"(y), "f"(z), "f"(w)
               : "memory");
}

__device__ __forceinline__ void st_global_cs_v4_f32(float* ptr,
                                                    float x,
                                                    float y,
                                                    float z,
                                                    float w) {
  asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "f"(x), "f"(y), "f"(z), "f"(w)
               : "memory");
}

__device__ __forceinline__ void cp_async_ca_shared_global_16(void* smem_ptr,
                                                             void const* gmem_ptr) {
  unsigned const smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_bulk_shared_global(void* smem_ptr,
                                                             void const* gmem_ptr,
                                                             uint32_t bytes,
                                                             uint64_t* bar) {
  unsigned const smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  unsigned const bar_addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
               "[%0], [%1], %2, [%3];"
               :
               : "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(bar_addr)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(PendingGroups) : "memory");
}

__device__ __forceinline__ void cluster_barrier_inline() {
  asm volatile("barrier.cluster.arrive;" ::: "memory");
  asm volatile("barrier.cluster.wait;" ::: "memory");
}

__device__ __forceinline__ void cluster_barrier_inline_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
  asm volatile("barrier.cluster.wait;" ::: "memory");
}

__device__ __forceinline__ void cluster_dsm_owner_lifetime_barrier(int rank) {
  asm volatile("barrier.cluster.arrive;" ::: "memory");
  if (rank == 0) {
    asm volatile("barrier.cluster.wait;" ::: "memory");
  }
}

__device__ __forceinline__ unsigned smem_addr_u32(void const* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ unsigned map_shared_rank_u32(void const* ptr, int rank) {
  unsigned mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(mapped)
               : "r"(smem_addr_u32(ptr)), "r"(rank));
  return mapped;
}

__device__ __forceinline__ void mbarrier_init_local(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(smem_addr_u32(bar)), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_init_fence_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx_local(uint64_t* bar,
                                                                uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(smem_addr_u32(bar)), "r"(bytes)
               : "memory");
}


__device__ __forceinline__ void mbarrier_wait_parity_local(uint64_t* bar,
                                                           uint32_t phase) {
  uint32_t done = 0;
  do {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.b32 %0, 1, 0, p;\n\t"
                 "}"
                 : "=r"(done)
                 : "r"(smem_addr_u32(bar)), "r"(phase)
                 : "memory");
  } while (!done);
}

__device__ __forceinline__ void st_async_shared_cluster_mbarrier_f32(
    float* ptr,
    float value,
    uint64_t* bar,
    int peer_rank) {
  unsigned const remote_ptr = map_shared_rank_u32(ptr, peer_rank);
  unsigned const remote_bar = map_shared_rank_u32(bar, peer_rank);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
               :
               : "r"(remote_ptr), "f"(value), "r"(remote_bar)
               : "memory");
}

__device__ __forceinline__ void st_shared_cluster_f32(float* ptr,
                                                       float value,
                                                       int peer_rank) {
  unsigned const remote_ptr = map_shared_rank_u32(ptr, peer_rank);
  asm volatile("st.shared::cluster.f32 [%0], %1;"
               :
               : "r"(remote_ptr), "f"(value)
               : "memory");
}

__device__ __forceinline__ float fma_ptx(float a, float b, float c) {
  float out;
  asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;"
               : "=f"(out)
               : "f"(a), "f"(b), "f"(c));
  return out;
}

__device__ __forceinline__ float rsqrt_ptx(float x) {
  float root;
  float out;
  asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(root) : "f"(x));
  asm volatile("rcp.rn.f32 %0, %1;" : "=f"(out) : "f"(root));
  return out;
}

__device__ __forceinline__ float rsqrt_fast_ptx(float x) {
  float out;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
  return out;
}

__device__ __forceinline__ float load_dtype(void const* ptr, DType dtype, int64_t idx) {
  switch (dtype) {
    case DType::kHalf:
      return __half2float(reinterpret_cast<half const*>(ptr)[idx]);
    case DType::kBFloat16:
      return __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(ptr)[idx]);
    case DType::kFloat:
      return reinterpret_cast<float const*>(ptr)[idx];
    default:
      return 0.0f;
  }
}

__device__ __forceinline__ void store_dtype(void* ptr, DType dtype, int64_t idx, float value) {
  switch (dtype) {
    case DType::kHalf:
      reinterpret_cast<half*>(ptr)[idx] = __float2half_rn(value);
      break;
    case DType::kBFloat16:
      reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16_rn(value);
      break;
    case DType::kFloat:
      reinterpret_cast<float*>(ptr)[idx] = value;
      break;
    default:
      break;
  }
}

__device__ __forceinline__ int64_t idx3(Tensor3 t, int64_t m, int64_t h, int64_t n) {
  return m * t.stride_m + h * t.stride_h + n * t.stride_n;
}

__device__ __forceinline__ int64_t idx2(Tensor2 t, int64_t m, int64_t h) {
  return m * t.stride_m + h * t.stride_h;
}

__device__ __forceinline__ int64_t idx_affine(Affine t, int64_t h, int64_t n) {
  return (t.per_head ? h * t.stride_h : 0) + n * t.stride_n;
}

template <int Width>
__device__ __forceinline__ float warp_reduce_sum_width(float value) {
  #pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum_full(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum_lane0_down(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float load_warp_uniform(float const* ptr, int64_t idx) {
  float value = 0.0f;
  if ((threadIdx.x & 31) == 0) {
    value = ptr[idx];
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

template <int ThreadsPerRow>
__device__ __forceinline__ float row_reduce_sum(float value, float* workspace) {
  int const tid = threadIdx.x;
  int const lane_in_row = tid % ThreadsPerRow;
  if constexpr (ThreadsPerRow <= 32) {
    return warp_reduce_sum_width<ThreadsPerRow>(value);
  } else {
    constexpr int kWarpsPerRow = ThreadsPerRow / 32;
    int const row_group = tid / ThreadsPerRow;
    int const lane = tid & 31;
    int const warp_in_row = lane_in_row >> 5;
    float warp_value = warp_reduce_sum_full(value);
    if (lane == 0) {
      workspace[row_group * kWarpsPerRow + warp_in_row] = warp_value;
    }
    __syncthreads();
    float row_value = lane < kWarpsPerRow ? workspace[row_group * kWarpsPerRow + lane] : 0.0f;
    return warp_reduce_sum_full(row_value);
  }
}

template <int ThreadsPerRow>
__device__ __forceinline__ float row_reduce_sum_lane0(float value, float* workspace) {
  int const tid = threadIdx.x;
  if constexpr (ThreadsPerRow <= 32) {
    return warp_reduce_sum_width<ThreadsPerRow>(value);
  } else {
    constexpr int kWarpsPerRow = ThreadsPerRow / 32;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    float warp_value = warp_reduce_sum_full(value);
    if (lane == 0) {
      workspace[warp] = warp_value;
    }
    __syncthreads();
    float row_value = tid < kWarpsPerRow ? workspace[tid] : 0.0f;
    if (warp == 0) {
      row_value = warp_reduce_sum_full(row_value);
    }
    return row_value;
  }
}

template <int ThreadsPerRow, bool SkipInitialBarrier = false>
__device__ __forceinline__ float row_reduce_sum_shared_broadcast(float value, float* workspace) {
  static_assert(ThreadsPerRow > 32);
  constexpr int kWarpsPerRow = ThreadsPerRow / 32;
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  float const warp_value = warp_reduce_sum_lane0_down(value);
  if (lane == 0) {
    workspace[warp] = warp_value;
  }
  if constexpr (!SkipInitialBarrier) {
    __syncthreads();
  }
  if constexpr (kWarpsPerRow == 4) {
    float total = 0.0f;
    if (lane == 0) {
      unsigned const addr = static_cast<unsigned>(__cvta_generic_to_shared(workspace));
      float a;
      float b;
      float c;
      float d;
      asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                   : "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                   : "r"(addr));
      total = (a + b) + (c + d);
    }
    return __shfl_sync(0xffffffffu, total, 0);
  } else {
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < kWarpsPerRow; ++i) {
      total += workspace[i];
    }
    return total;
  }
}


template <typename T>
__device__ __forceinline__ T* align_t_ptr(unsigned char*& ptr) {
  uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
  raw = (raw + alignof(T) - 1) & ~(uintptr_t(alignof(T)) - 1);
  ptr = reinterpret_cast<unsigned char*>(raw);
  return reinterpret_cast<T*>(ptr);
}

template <int ThreadsPerRow, int NumThreads>
__global__ void rmsnorm_fwd_kernel(
    Tensor3 x,
    Affine weight,
    Affine bias,
    Tensor3 residual,
    Tensor3 out,
    Tensor3 residual_out,
    Tensor2 rstd_out,
    int64_t rows_m,
    int64_t heads,
    int64_t N,
    float eps) {
  constexpr int kRowsPerBlock = NumThreads / ThreadsPerRow;
  constexpr int kWarpsPerRow = (ThreadsPerRow + 31) / 32;

  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const tid = threadIdx.x;
  int const row_group = tid / ThreadsPerRow;
  int const lane = tid - row_group * ThreadsPerRow;
  int64_t const flat_row = int64_t(blockIdx.x) * kRowsPerBlock + row_group;
  int64_t const total_rows = rows_m * heads;
  int64_t const m = flat_row / heads;
  int64_t const h = flat_row - m * heads;
  bool const valid_row = flat_row < total_rows;

  float thread_sum_sq = 0.0f;
  if (valid_row) {
    for (int64_t col = lane; col < N; col += ThreadsPerRow) {
      float value = load_dtype(x.ptr, x.dtype, idx3(x, m, h, col));
      if (residual.dtype != DType::kNone) {
        value += load_dtype(residual.ptr, residual.dtype, idx3(residual, m, h, col));
      }
      thread_sum_sq = fma_ptx(value, value, thread_sum_sq);
    }
  }

  float const sum_sq = row_reduce_sum<ThreadsPerRow>(thread_sum_sq, workspace);
  float const variance = sum_sq * (1.0f / static_cast<float>(N)) + eps;
  float const rstd = rsqrt_ptx(variance);

  if (valid_row && lane == 0 && rstd_out.dtype != DType::kNone) {
    store_dtype(rstd_out.ptr, rstd_out.dtype, idx2(rstd_out, m, h), rstd);
  }

  if (valid_row) {
    for (int64_t col = lane; col < N; col += ThreadsPerRow) {
      float value = load_dtype(x.ptr, x.dtype, idx3(x, m, h, col));
      if (residual.dtype != DType::kNone) {
        value += load_dtype(residual.ptr, residual.dtype, idx3(residual, m, h, col));
      }
      if (residual_out.dtype != DType::kNone) {
        store_dtype(residual_out.ptr, residual_out.dtype, idx3(residual_out, m, h, col), value);
      }
      float y = value * rstd;
      if (weight.dtype != DType::kNone) {
        y *= load_dtype(weight.ptr, weight.dtype, idx_affine(weight, h, col));
      }
      if (bias.dtype != DType::kNone) {
        y += load_dtype(bias.ptr, bias.dtype, idx_affine(bias, h, col));
      }
      store_dtype(out.ptr, out.dtype, idx3(out, m, h, col), y);
    }
  }
}

template <typename T,
          int ThreadsPerRow,
          int NumThreads,
          int MaxVecs,
          bool HasResidual,
          bool HasRstd,
          bool PreloadWeight = false>
__global__ void rmsnorm_fwd_contig_kernel(
    T const* __restrict__ x,
    float const* __restrict__ weight,
    float const* __restrict__ bias,
    T const* __restrict__ residual,
    T* __restrict__ out,
    T* __restrict__ residual_out,
    float* __restrict__ rstd_out,
    int64_t M,
    int64_t N,
    float eps) {
  constexpr int kVec = Vec128<T>::kElements;
  constexpr int kRowsPerBlock = NumThreads / ThreadsPerRow;
  constexpr int kWarpsPerRow = (ThreadsPerRow + 31) / 32;

  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const tid = threadIdx.x;
  int const row_group = tid / ThreadsPerRow;
  int const lane = tid - row_group * ThreadsPerRow;
  int64_t const row = int64_t(blockIdx.x) * kRowsPerBlock + row_group;
  bool const valid_row = row < M;

  uint4 raw_x[MaxVecs];
  uint4 raw_res[MaxVecs];
  static_assert(!PreloadWeight || HasResidual);
  Float4Raw weight_cache[PreloadWeight ? MaxVecs * (sizeof(T) == 2 ? 2 : 1) : 1];
  float thread_sum_sq = 0.0f;

  if (valid_row) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int const vec = lane + i * ThreadsPerRow;
      int64_t const col_base = int64_t(vec) * kVec;
      raw_x[i] = ld_global_u128(x + row * N + col_base);
      if constexpr (HasResidual) {
        raw_res[i] = ld_global_cg_u128(residual + row * N + col_base);
        if constexpr (PreloadWeight) {
          weight_cache[i * (sizeof(T) == 2 ? 2 : 1)].raw = ld_global_ca_u128(weight + col_base);
          if constexpr (sizeof(T) == 2) {
            weight_cache[i * 2 + 1].raw = ld_global_ca_u128(weight + col_base + 4);
          }
        }
      }
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sr;
      sx.raw = raw_x[i];
      if constexpr (HasResidual) {
        sr.raw = raw_res[i];
      }
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float value = to_float_t<T>(sx.elem[j]);
        if constexpr (HasResidual) {
          value += to_float_t<T>(sr.elem[j]);
        }
        thread_sum_sq = fma_ptx(value, value, thread_sum_sq);
      }
    }
  }

  float const sum_sq = row_reduce_sum<ThreadsPerRow>(thread_sum_sq, workspace);
  float const variance = sum_sq * (1.0f / static_cast<float>(N)) + eps;
  float const rstd = rsqrt_fast_ptx(variance);

  if constexpr (HasRstd) {
    if (valid_row && lane == 0) {
      rstd_out[row] = rstd;
    }
  }

  if (valid_row) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int const vec = lane + i * ThreadsPerRow;
      int64_t const col_base = int64_t(vec) * kVec;
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sr;
      typename Vec128<T>::Storage sy;
      typename Vec128<T>::Storage sro;
      Float4Raw w0;
      Float4Raw w1;
      sx.raw = raw_x[i];
      if constexpr (HasResidual) {
        sr.raw = raw_res[i];
      }
      if constexpr (PreloadWeight) {
        w0 = weight_cache[i * (sizeof(T) == 2 ? 2 : 1)];
        if constexpr (sizeof(T) == 2) {
          w1 = weight_cache[i * 2 + 1];
        }
      } else if constexpr (sizeof(T) == 2) {
        w0.raw = ld_global_ca_u128(weight + col_base);
        w1.raw = ld_global_ca_u128(weight + col_base + 4);
      } else {
        w0.raw = ld_global_ca_u128(weight + col_base);
      }
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        int64_t const col = col_base + j;
        float value = to_float_t<T>(sx.elem[j]);
        if constexpr (HasResidual) {
          value += to_float_t<T>(sr.elem[j]);
          sro.elem[j] = from_float_t<T>(value);
        }
        float const w = sizeof(T) == 2 ? (j < 4 ? w0.elem[j] : w1.elem[j - 4]) : w0.elem[j];
        float y = value * rstd * w;
        if (bias != nullptr) {
          y += bias[col];
        }
        sy.elem[j] = from_float_t<T>(y);
      }
      if constexpr (HasResidual) {
        st_global_u128(residual_out + row * N + col_base, sro.raw);
      }
      st_global_u128(out + row * N + col_base, sy.raw);
    }
  }
}

template <int ThreadsPerRow, int NumThreads, int MaxVecs>
__global__ __launch_bounds__(NumThreads, 1)
void rmsnorm_fwd_contig_fp32_preload_w_kernel(
    float const* __restrict__ x,
    float const* __restrict__ weight,
    float* __restrict__ out,
    int64_t M,
    int64_t N,
    float eps) {
  constexpr int kVec = Vec128<float>::kElements;
  constexpr int kRowsPerBlock = NumThreads / ThreadsPerRow;

  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const tid = threadIdx.x;
  int const row_group = tid / ThreadsPerRow;
  int const lane = tid - row_group * ThreadsPerRow;
  int64_t const row = int64_t(blockIdx.x) * kRowsPerBlock + row_group;
  bool const valid_row = row < M;

  uint4 raw_x[MaxVecs];
  Float4Raw sw[MaxVecs];
  float thread_sum_sq = 0.0f;

  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int const vec = lane + i * ThreadsPerRow;
    int64_t const col_base = int64_t(vec) * kVec;
    sw[i].raw = ld_global_ca_u128(weight + col_base);
    if (valid_row) {
      raw_x[i] = ld_global_cg_u128(x + row * N + col_base);
      Float4Raw sx;
      sx.raw = raw_x[i];
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        thread_sum_sq = fma_ptx(sx.elem[j], sx.elem[j], thread_sum_sq);
      }
    }
  }

  float const sum_sq = row_reduce_sum<ThreadsPerRow>(thread_sum_sq, workspace);
  float const variance = sum_sq * (1.0f / static_cast<float>(N)) + eps;
  float const rstd = rsqrt_fast_ptx(variance);

  if (valid_row) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int const vec = lane + i * ThreadsPerRow;
      int64_t const col_base = int64_t(vec) * kVec;
      Float4Raw sx;
      sx.raw = raw_x[i];
      st_global_v4_f32(out + row * N + col_base,
                       sx.elem[0] * rstd * sw[i].elem[0],
                       sx.elem[1] * rstd * sw[i].elem[1],
                       sx.elem[2] * rstd * sw[i].elem[2],
                       sx.elem[3] * rstd * sw[i].elem[3]);
    }
  }
}


template <int ThreadsPerRow, int MaxVecs>
__global__ __maxnreg__(142)
void rmsnorm_fwd_contig_fp32_smem_async_preload_w_kernel(
    float const* __restrict__ x,
    float const* __restrict__ weight,
    float* __restrict__ out,
    float eps) {
  constexpr int kVec = Vec128<float>::kElements;
  constexpr int kStageValues = ThreadsPerRow * MaxVecs;
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  uint4* x_smem = align_t_ptr<uint4>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(x_smem + kStageValues);
  float* workspace = align_t_ptr<float>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(workspace + ThreadsPerRow / 32);
  uint64_t* bulk_bar = align_t_ptr<uint64_t>(smem_ptr);

  int const lane = threadIdx.x;
  float const* row_x = x + (int64_t(blockIdx.x) << 13);
  float* row_out = out + (int64_t(blockIdx.x) << 13);
  if (lane == 0) {
    mbarrier_init_local(bulk_bar, 1);
    mbarrier_init_fence_cluster();
  }
  __syncthreads();
  if (lane == 0) {
    constexpr uint32_t kBytes = kStageValues * sizeof(uint4);
    mbarrier_arrive_expect_tx_local(bulk_bar, kBytes);
    cp_async_bulk_shared_global(x_smem, row_x, kBytes, bulk_bar);
  }
  if ((lane & 31) == 0) {
    mbarrier_wait_parity_local(bulk_bar, 0);
  }
  __syncwarp();

  uint4 raw_x[MaxVecs];
  float sum_sq0 = 0.0f;
  float sum_sq1 = 0.0f;
  float sum_sq2 = 0.0f;
  float sum_sq3 = 0.0f;
  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int const vec = lane + i * ThreadsPerRow;
    raw_x[i] = x_smem[vec];
    Float4Raw sx;
    sx.raw = raw_x[i];
    sum_sq0 = fma_ptx(sx.elem[0], sx.elem[0], sum_sq0);
    sum_sq1 = fma_ptx(sx.elem[1], sx.elem[1], sum_sq1);
    sum_sq2 = fma_ptx(sx.elem[2], sx.elem[2], sum_sq2);
    sum_sq3 = fma_ptx(sx.elem[3], sx.elem[3], sum_sq3);
  }

  float const thread_sum_sq = (sum_sq0 + sum_sq1) + (sum_sq2 + sum_sq3);
  float const sum_sq = row_reduce_sum_shared_broadcast<ThreadsPerRow>(thread_sum_sq, workspace);
  float const rstd = rsqrt_fast_ptx(sum_sq * (1.0f / 8192.0f) + eps);
  uint64_t weight_policy = 0;
  if ((lane & 31) == 0) {
    weight_policy = createpolicy_evict_last_l2(weight, 8192 * sizeof(float));
  }
  weight_policy = __shfl_sync(0xffffffff, weight_policy, 0);

  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int const vec = lane + i * ThreadsPerRow;
    int const col_base = vec * kVec;
    Float4Raw sx;
    Float4Raw sw;
    sx.raw = raw_x[i];
    sw.raw = ld_global_nc_cache_hint_l2_256_u128(weight + col_base, weight_policy);
    st_global_v4_f32(row_out + col_base,
                     sx.elem[0] * rstd * sw.elem[0],
                     sx.elem[1] * rstd * sw.elem[1],
                     sx.elem[2] * rstd * sw.elem[2],
                     sx.elem[3] * rstd * sw.elem[3]);
  }
}


template <typename T, int ThreadsPerRow, int MaxVecs>
__global__ void rmsnorm_fwd_contig_stream_kernel(
    T const* __restrict__ x,
    float const* __restrict__ weight,
    T* __restrict__ out,
    int64_t M,
    int64_t N,
    float eps) {
  constexpr int kVec = Vec128<T>::kElements;
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const lane = threadIdx.x;
  int64_t const row = int64_t(blockIdx.x);
  bool const valid_row = row < M;
  float thread_sum_sq = 0.0f;

  if (valid_row) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
      uint4 const raw = ld_global_u128(x + row * N + col_base);
      typename Vec128<T>::Storage sx;
      sx.raw = raw;
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const value = to_float_t<T>(sx.elem[j]);
        thread_sum_sq = fma_ptx(value, value, thread_sum_sq);
      }
    }
  }

  float const sum_sq = row_reduce_sum<ThreadsPerRow>(thread_sum_sq, workspace);
  float const variance = sum_sq * (1.0f / static_cast<float>(N)) + eps;
  float const rstd = rsqrt_fast_ptx(variance);

  if (valid_row) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
      uint4 const raw = ld_global_u128(x + row * N + col_base);
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sy;
      Float4Raw w0;
      Float4Raw w1;
      sx.raw = raw;
      if constexpr (sizeof(T) == 2) {
        w0.raw = ld_global_u128(weight + col_base);
        w1.raw = ld_global_u128(weight + col_base + 4);
      } else {
        w0.raw = ld_global_u128(weight + col_base);
      }
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const value = to_float_t<T>(sx.elem[j]);
        float const w = sizeof(T) == 2 ? (j < 4 ? w0.elem[j] : w1.elem[j - 4]) : w0.elem[j];
        sy.elem[j] = from_float_t<T>(value * rstd * w);
      }
      st_global_u128(out + row * N + col_base, sy.raw);
    }
  }
}

template <typename T,
          int ClusterN,
          int ThreadsPerRow,
          int MaxVecs,
          bool CacheXSmem,
          int RegCacheVecs,
          int TailSmemStartVec = 0>
__global__ void rmsnorm_fwd_cluster_kernel(
    T const* __restrict__ x,
    float const* __restrict__ weight,
    T* __restrict__ out,
    int64_t M,
    int64_t N,
    float eps) {
  static_assert(RegCacheVecs >= 0 && RegCacheVecs <= MaxVecs);
  static_assert(TailSmemStartVec >= 0 && TailSmemStartVec <= MaxVecs);
  static_assert(TailSmemStartVec == 0 || RegCacheVecs <= TailSmemStartVec);
  constexpr int kVec = Vec128<T>::kElements;
  constexpr int64_t kSegmentCols = int64_t(ThreadsPerRow) * kVec * MaxVecs;
  constexpr bool kTailSmem = TailSmemStartVec > 0;
  constexpr int kTailSmemVecs = kTailSmem ? (MaxVecs - TailSmemStartVec) : 0;
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  uint4* x_smem = nullptr;
  if constexpr (CacheXSmem || kTailSmem) {
    x_smem = align_t_ptr<uint4>(smem_ptr);
    int const smem_vecs = CacheXSmem ? MaxVecs : kTailSmemVecs;
    smem_ptr = reinterpret_cast<unsigned char*>(x_smem + ThreadsPerRow * smem_vecs);
  }
  float* workspace = align_t_ptr<float>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(workspace + 32);
  float* cluster_sums = align_t_ptr<float>(smem_ptr);

  cg::cluster_group cluster = cg::this_cluster();
  int const rank = int(cluster.block_rank());
  int const lane = threadIdx.x;
  int64_t const row = int64_t(blockIdx.x) / ClusterN;
  int64_t const segment_base = int64_t(rank) * kSegmentCols;
  bool const valid_row = row < M;
  uint4 raw_x_cache[RegCacheVecs > 0 ? RegCacheVecs : 1];

  float thread_sum = 0.0f;
  if (valid_row) {
    if constexpr (CacheXSmem) {
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
        cp_async_ca_shared_global_16(x_smem + lane + i * ThreadsPerRow,
                                     x + row * N + col_base);
      }
      cp_async_commit_group();
      cp_async_wait_group<0>();
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
        uint4 const raw = x_smem[lane + i * ThreadsPerRow];
        if constexpr (RegCacheVecs > 0) {
          if (i < RegCacheVecs) {
            raw_x_cache[i] = raw;
          }
        }
        typename Vec128<T>::Storage sx;
        sx.raw = raw;
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const value = to_float_t<T>(sx.elem[j]);
          thread_sum = fma_ptx(value, value, thread_sum);
        }
      }
    } else if constexpr (kTailSmem) {
      // For very wide fp32 rows, keep the low-register prefix in registers and
      // stage only the tail fragment. This avoids most of the second x read
      // without the occupancy loss of caching the whole CTA tile in smem.
      #pragma unroll
      for (int i = TailSmemStartVec; i < MaxVecs; ++i) {
        int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
        int const smem_vec = lane + (i - TailSmemStartVec) * ThreadsPerRow;
        cp_async_ca_shared_global_16(x_smem + smem_vec, x + row * N + col_base);
      }
      cp_async_commit_group();
      #pragma unroll
      for (int i = 0; i < TailSmemStartVec; ++i) {
        int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
        uint4 const raw = ld_global_cg_u128(x + row * N + col_base);
        if constexpr (RegCacheVecs > 0) {
          if (i < RegCacheVecs) {
            raw_x_cache[i] = raw;
          }
        }
        typename Vec128<T>::Storage sx;
        sx.raw = raw;
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const value = to_float_t<T>(sx.elem[j]);
          thread_sum = fma_ptx(value, value, thread_sum);
        }
      }
      cp_async_wait_group<0>();
      #pragma unroll
      for (int i = TailSmemStartVec; i < MaxVecs; ++i) {
        int const smem_vec = lane + (i - TailSmemStartVec) * ThreadsPerRow;
        uint4 const raw = x_smem[smem_vec];
        typename Vec128<T>::Storage sx;
        sx.raw = raw;
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const value = to_float_t<T>(sx.elem[j]);
          thread_sum = fma_ptx(value, value, thread_sum);
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
        uint4 const raw = ld_global_cg_u128(x + row * N + col_base);
        if constexpr (RegCacheVecs > 0) {
          if (i < RegCacheVecs) {
            raw_x_cache[i] = raw;
          }
        }
        typename Vec128<T>::Storage sx;
        sx.raw = raw;
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const value = to_float_t<T>(sx.elem[j]);
          thread_sum = fma_ptx(value, value, thread_sum);
        }
      }
    }
  }

  float const slice_sum = row_reduce_sum<ThreadsPerRow>(thread_sum, workspace);
  if (lane < ClusterN) {
    st_shared_cluster_f32(cluster_sums + rank, slice_sum, lane);
  }
  cluster_barrier_inline();

  float total = 0.0f;
  #pragma unroll
  for (int i = 0; i < ClusterN; ++i) {
    total += cluster_sums[i];
  }
  float const variance = total * (1.0f / static_cast<float>(N)) + eps;
  float const rstd = rsqrt_fast_ptx(variance);

  if (valid_row) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
      uint4 raw_x;
      if constexpr (RegCacheVecs > 0) {
        if (i < RegCacheVecs) {
          raw_x = raw_x_cache[i];
        } else if constexpr (CacheXSmem) {
          raw_x = x_smem[lane + i * ThreadsPerRow];
        } else if constexpr (kTailSmem) {
          if (i >= TailSmemStartVec) {
            raw_x = x_smem[lane + (i - TailSmemStartVec) * ThreadsPerRow];
          } else {
            raw_x = ld_global_cg_u128(x + row * N + col_base);
          }
        } else {
          raw_x = ld_global_u128(x + row * N + col_base);
        }
      } else if constexpr (CacheXSmem) {
        raw_x = x_smem[lane + i * ThreadsPerRow];
      } else if constexpr (kTailSmem) {
        if (i >= TailSmemStartVec) {
          raw_x = x_smem[lane + (i - TailSmemStartVec) * ThreadsPerRow];
        } else {
          raw_x = ld_global_cg_u128(x + row * N + col_base);
        }
      } else {
        raw_x = ld_global_u128(x + row * N + col_base);
      }
      typename Vec128<T>::Storage sx;
      Float4Raw w0;
      Float4Raw w1;
      sx.raw = raw_x;
      if constexpr (sizeof(T) == 2) {
        w0.raw = ld_global_ca_u128(weight + col_base);
        w1.raw = ld_global_ca_u128(weight + col_base + 4);
      } else {
        w0.raw = ld_global_ca_u128(weight + col_base);
      }
      if constexpr (sizeof(T) == 4) {
        st_global_cs_v4_f32(out + row * N + col_base,
                            sx.elem[0] * rstd * w0.elem[0],
                            sx.elem[1] * rstd * w0.elem[1],
                            sx.elem[2] * rstd * w0.elem[2],
                            sx.elem[3] * rstd * w0.elem[3]);
      } else {
        typename Vec128<T>::Storage sy;
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const value = to_float_t<T>(sx.elem[j]);
          float const w = j < 4 ? w0.elem[j] : w1.elem[j - 4];
          sy.elem[j] = from_float_t<T>(value * rstd * w);
        }
        st_global_u128(out + row * N + col_base, sy.raw);
      }
    }
  }
}

template <typename T, int ThreadsPerRow, int MaxVecs, bool HasResidual = false>
__global__ void rmsnorm_fwd_split_sumsq_kernel(
    T const* __restrict__ x,
    T const* __restrict__ residual,
    float* __restrict__ sumsq_partial,
    int64_t M,
    int64_t N,
    int64_t slices,
    int64_t segment_cols) {
  constexpr int kVec = Vec128<T>::kElements;
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const lane = threadIdx.x;
  int64_t const row = int64_t(blockIdx.x);
  int64_t const slice = int64_t(blockIdx.y);
  int64_t const segment_base = slice * segment_cols;

  float thread_sum = 0.0f;
  if (row < M) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
      uint4 const raw = ld_global_u128(x + row * N + col_base);
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sr;
      sx.raw = raw;
      if constexpr (HasResidual) {
        sr.raw = ld_global_u128(residual + row * N + col_base);
      }
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float value = to_float_t<T>(sx.elem[j]);
        if constexpr (HasResidual) {
          value += to_float_t<T>(sr.elem[j]);
        }
        thread_sum = fma_ptx(value, value, thread_sum);
      }
    }
  }

  float const sum = row_reduce_sum<ThreadsPerRow>(thread_sum, workspace);
  if (lane == 0 && row < M) {
    sumsq_partial[row * slices + slice] = sum;
  }
}

__global__ void rmsnorm_fwd_reduce_sumsq_kernel(
    float const* __restrict__ sumsq_partial,
    float* __restrict__ rstd,
    int64_t M,
    int64_t N,
    int64_t slices,
    float eps) {
  int64_t const row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= M) {
    return;
  }
  float sum = 0.0f;
  for (int64_t s = 0; s < slices; ++s) {
    sum += sumsq_partial[row * slices + s];
  }
  float const variance = sum * (1.0f / static_cast<float>(N)) + eps;
  rstd[row] = rsqrt_fast_ptx(variance);
}

template <typename T, int ThreadsPerRow, int MaxVecs, bool HasResidual = false>
__global__ void rmsnorm_fwd_split_output_kernel(
    T const* __restrict__ x,
    float const* __restrict__ weight,
    T const* __restrict__ residual,
    float const* __restrict__ rstd,
    T* __restrict__ out,
    T* __restrict__ residual_out,
    int64_t M,
    int64_t N,
    int64_t segment_cols) {
  constexpr int kVec = Vec128<T>::kElements;
  int const lane = threadIdx.x;
  int64_t const row = int64_t(blockIdx.x);
  int64_t const slice = int64_t(blockIdx.y);
  int64_t const segment_base = slice * segment_cols;

  if (row >= M) {
    return;
  }
  float const row_rstd = rstd[row];
  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
    uint4 const raw_x = ld_global_u128(x + row * N + col_base);
    typename Vec128<T>::Storage sx;
    typename Vec128<T>::Storage sr;
    typename Vec128<T>::Storage sy;
    typename Vec128<T>::Storage sres_out;
    Float4Raw w0;
    Float4Raw w1;
    sx.raw = raw_x;
    if constexpr (HasResidual) {
      sr.raw = ld_global_u128(residual + row * N + col_base);
    }
    if constexpr (sizeof(T) == 2) {
      w0.raw = ld_global_u128(weight + col_base);
      w1.raw = ld_global_u128(weight + col_base + 4);
    } else {
      w0.raw = ld_global_u128(weight + col_base);
    }
    #pragma unroll
    for (int j = 0; j < kVec; ++j) {
      float value = to_float_t<T>(sx.elem[j]);
      if constexpr (HasResidual) {
        value += to_float_t<T>(sr.elem[j]);
        sres_out.elem[j] = from_float_t<T>(value);
      }
      float const w = sizeof(T) == 2 ? (j < 4 ? w0.elem[j] : w1.elem[j - 4]) : w0.elem[j];
      sy.elem[j] = from_float_t<T>(value * row_rstd * w);
    }
    if constexpr (HasResidual) {
      st_global_u128(residual_out + row * N + col_base, sres_out.raw);
    }
    st_global_u128(out + row * N + col_base, sy.raw);
  }
}

template <int ThreadsPerRow, int NumThreads>
__global__ void rmsnorm_bwd_kernel(
    Tensor3 x,
    Affine weight,
    Tensor3 dout,
    Tensor2 rstd_in,
    Tensor3 dresidual_out,
    Tensor3 dx,
    float* __restrict__ dw,
    float* __restrict__ db,
    Tensor3 dresidual,
    int64_t rows_m,
    int64_t heads,
    int64_t N) {
  constexpr int kRowsPerBlock = NumThreads / ThreadsPerRow;

  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const tid = threadIdx.x;
  int const row_group = tid / ThreadsPerRow;
  int const lane = tid - row_group * ThreadsPerRow;
  int64_t const flat_row = int64_t(blockIdx.x) * kRowsPerBlock + row_group;
  int64_t const total_rows = rows_m * heads;
  int64_t const m = flat_row / heads;
  int64_t const h = flat_row - m * heads;
  bool const valid_row = flat_row < total_rows;

  float rstd = 0.0f;
  if (valid_row) {
    rstd = load_dtype(rstd_in.ptr, rstd_in.dtype, idx2(rstd_in, m, h));
  }

  float thread_sum = 0.0f;
  if (valid_row) {
    for (int64_t col = lane; col < N; col += ThreadsPerRow) {
      float const x_val = load_dtype(x.ptr, x.dtype, idx3(x, m, h, col));
      float const x_hat = x_val * rstd;
      float const dout_val = load_dtype(dout.ptr, dout.dtype, idx3(dout, m, h, col));
      float wdy = dout_val;
      if (weight.dtype != DType::kNone) {
        wdy *= load_dtype(weight.ptr, weight.dtype, idx_affine(weight, h, col));
      }
      thread_sum = fma_ptx(x_hat, wdy, thread_sum);
    }
  }

  float const row_sum = row_reduce_sum<ThreadsPerRow>(thread_sum, workspace);
  float const mean = row_sum * (1.0f / static_cast<float>(N));
  float const mean_rstd = mean * rstd;

  if (valid_row) {
    for (int64_t col = lane; col < N; col += ThreadsPerRow) {
      float const x_val = load_dtype(x.ptr, x.dtype, idx3(x, m, h, col));
      float const x_hat = x_val * rstd;
      float const dout_val = load_dtype(dout.ptr, dout.dtype, idx3(dout, m, h, col));
      float wdy = dout_val;
      if (weight.dtype != DType::kNone) {
        wdy *= load_dtype(weight.ptr, weight.dtype, idx_affine(weight, h, col));
      }
      float grad = fma_ptx(-x_hat, mean_rstd, wdy * rstd);
      if (dresidual_out.dtype != DType::kNone) {
        grad += load_dtype(dresidual_out.ptr, dresidual_out.dtype, idx3(dresidual_out, m, h, col));
      }
      store_dtype(dx.ptr, dx.dtype, idx3(dx, m, h, col), grad);
      if (dresidual.dtype != DType::kNone) {
        store_dtype(dresidual.ptr, dresidual.dtype, idx3(dresidual, m, h, col), grad);
      }
    }
  }
}

__device__ __forceinline__ float block_reduce_sum(float value, float* workspace) {
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  value = warp_reduce_sum_full(value);
  if (lane == 0) {
    workspace[warp] = value;
  }
  __syncthreads();
  float block_value = threadIdx.x < (blockDim.x >> 5) ? workspace[lane] : 0.0f;
  if (warp == 0) {
    block_value = warp_reduce_sum_full(block_value);
  }
  return block_value;
}

__global__ void rmsnorm_affine_grad_kernel(
    Tensor3 x,
    Tensor3 dout,
    Tensor2 rstd_in,
    float* __restrict__ dw,
    float* __restrict__ db,
    int64_t rows_m,
    int64_t heads,
    int64_t N) {
  extern __shared__ float workspace[];
  int64_t const col = int64_t(blockIdx.x);
  int64_t const h = int64_t(blockIdx.y);

  float thread_dw = 0.0f;
  float thread_db = 0.0f;
  for (int64_t m = threadIdx.x; m < rows_m; m += blockDim.x) {
    float const rstd = load_dtype(rstd_in.ptr, rstd_in.dtype, idx2(rstd_in, m, h));
    float const x_hat = load_dtype(x.ptr, x.dtype, idx3(x, m, h, col)) * rstd;
    float const dout_val = load_dtype(dout.ptr, dout.dtype, idx3(dout, m, h, col));
    thread_dw = fma_ptx(dout_val, x_hat, thread_dw);
    thread_db += dout_val;
  }

  float sum_dw = 0.0f;
  if (dw != nullptr) {
    sum_dw = block_reduce_sum(thread_dw, workspace);
  }
  float sum_db = 0.0f;
  if (db != nullptr) {
    sum_db = block_reduce_sum(thread_db, workspace);
  }
  if (threadIdx.x == 0) {
    if (dw != nullptr) {
      dw[h * N + col] = sum_dw;
    }
    if (db != nullptr) {
      db[h * N + col] = sum_db;
    }
  }
}

template <typename T,
          int ThreadsPerRow,
          int MaxVecs,
          int StaticN = 0,
          bool HasDresidualOut = false>
__global__
void rmsnorm_bwd_contig_partial_kernel(
    T const* __restrict__ x,
    float const* __restrict__ weight,
    T const* __restrict__ dout,
    float const* __restrict__ rstd_in,
    T* __restrict__ dx,
    T const* __restrict__ dresidual_out,
    float* __restrict__ dw_partial,
    int64_t M,
    int64_t N) {
  constexpr int kVec = Vec128<T>::kElements;
  constexpr bool kHasStaticN = StaticN > 0;
  int64_t const row_stride = kHasStaticN ? int64_t(StaticN) : N;
  float const inv_n = kHasStaticN ? (1.0f / static_cast<float>(StaticN))
                                  : (1.0f / static_cast<float>(N));
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const lane = threadIdx.x;
  int64_t const partial_id = int64_t(blockIdx.x);
  int64_t const partial_count = int64_t(gridDim.x);

  Float4Raw w0[MaxVecs];
  Float4Raw w1[MaxVecs];
  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
    if constexpr (sizeof(T) == 2) {
      w0[i].raw = ld_global_ca_u128(weight + col_base);
      w1[i].raw = ld_global_ca_u128(weight + col_base + 4);
    } else {
      w0[i].raw = ld_global_ca_u128(weight + col_base);
    }
  }

  float dw_accum[MaxVecs * kVec];
  #pragma unroll
  for (int i = 0; i < MaxVecs * kVec; ++i) {
    dw_accum[i] = 0.0f;
  }

  if constexpr (MaxVecs <= 2 || (ThreadsPerRow == 256 && MaxVecs <= 4) ||
                (sizeof(T) == 4 && ThreadsPerRow == 128 && (MaxVecs == 4 || MaxVecs == 8))) {
    uint4 raw_x[MaxVecs];
    uint4 raw_dout[MaxVecs];
    int64_t row = partial_id;
    bool valid_row = row < M;
    float rstd = 0.0f;
    if (valid_row) {
      if constexpr (sizeof(T) == 2) {
        rstd = load_warp_uniform(rstd_in, row);
      } else {
        rstd = rstd_in[row];
      }
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
        if constexpr (sizeof(T) == 2 && StaticN == 8192) {
          raw_x[i] = ld_global_cg_u128(x + row * row_stride + col_base);
          raw_dout[i] = ld_global_cg_u128(dout + row * row_stride + col_base);
        } else {
          raw_x[i] = ld_global_u128(x + row * row_stride + col_base);
          raw_dout[i] = ld_global_u128(dout + row * row_stride + col_base);
        }
      }
    }

    while (valid_row) {
      float thread_dot = 0.0f;
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        typename Vec128<T>::Storage sx;
        typename Vec128<T>::Storage sdout;
        sx.raw = raw_x[i];
        sdout.raw = raw_dout[i];
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
          float const dout_val = to_float_t<T>(sdout.elem[j]);
          float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                         : w0[i].elem[j];
          thread_dot = fma_ptx(x_hat, dout_val * w, thread_dot);
        }
      }

      uint4 next_x[MaxVecs];
      uint4 next_dout[MaxVecs];
      int64_t const next_row = row + partial_count;
      bool const has_next = next_row < M;
      float next_rstd = 0.0f;
      if (has_next) {
        if constexpr (sizeof(T) == 2) {
          next_rstd = load_warp_uniform(rstd_in, next_row);
        } else {
          next_rstd = rstd_in[next_row];
        }
        #pragma unroll
        for (int i = 0; i < MaxVecs; ++i) {
          int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
          if constexpr (sizeof(T) == 2 && StaticN == 8192) {
            next_x[i] = ld_global_cg_u128(x + next_row * row_stride + col_base);
            next_dout[i] = ld_global_cg_u128(dout + next_row * row_stride + col_base);
          } else {
            next_x[i] = ld_global_u128(x + next_row * row_stride + col_base);
            next_dout[i] = ld_global_u128(dout + next_row * row_stride + col_base);
          }
        }
      }

      float const row_dot = row_reduce_sum<ThreadsPerRow>(thread_dot, workspace);
      float const mean = row_dot * inv_n;
      float const mean_rstd = mean * rstd;

      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
        typename Vec128<T>::Storage sx;
        typename Vec128<T>::Storage sdout;
        typename Vec128<T>::Storage sdx;
        typename Vec128<T>::Storage sdres;
        sx.raw = raw_x[i];
        sdout.raw = raw_dout[i];
        if constexpr (HasDresidualOut) {
          sdres.raw = ld_global_cg_u128(dresidual_out + row * row_stride + col_base);
        }
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
          float const dout_val = to_float_t<T>(sdout.elem[j]);
          float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                         : w0[i].elem[j];
          float const wdy = dout_val * w;
          float grad = fma_ptx(-x_hat, mean_rstd, wdy * rstd);
          if constexpr (HasDresidualOut) {
            grad += to_float_t<T>(sdres.elem[j]);
          }
          sdx.elem[j] = from_float_t<T>(grad);
          dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
        }
        st_global_cs_u128(dx + row * row_stride + col_base, sdx.raw);
      }

      if (has_next) {
        #pragma unroll
        for (int i = 0; i < MaxVecs; ++i) {
          raw_x[i] = next_x[i];
          raw_dout[i] = next_dout[i];
        }
      }
      row = next_row;
      rstd = next_rstd;
      valid_row = has_next;
    }
  } else {
    for (int64_t row = partial_id; row < M; row += partial_count) {
      uint4 raw_x[MaxVecs];
      uint4 raw_dout[MaxVecs];
      float rstd;
      if constexpr (sizeof(T) == 2) {
        rstd = load_warp_uniform(rstd_in, row);
      } else {
        rstd = rstd_in[row];
      }
      float thread_dot = 0.0f;

      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
        raw_x[i] = ld_global_u128(x + row * row_stride + col_base);
        raw_dout[i] = ld_global_u128(dout + row * row_stride + col_base);
        typename Vec128<T>::Storage sx;
        typename Vec128<T>::Storage sdout;
        sx.raw = raw_x[i];
        sdout.raw = raw_dout[i];
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
          float const dout_val = to_float_t<T>(sdout.elem[j]);
          float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                         : w0[i].elem[j];
          thread_dot = fma_ptx(x_hat, dout_val * w, thread_dot);
        }
      }

      float const row_dot = row_reduce_sum<ThreadsPerRow>(thread_dot, workspace);
      float const mean = row_dot * inv_n;
      float const mean_rstd = mean * rstd;

      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
        typename Vec128<T>::Storage sx;
        typename Vec128<T>::Storage sdout;
        typename Vec128<T>::Storage sdx;
        typename Vec128<T>::Storage sdres;
        sx.raw = raw_x[i];
        sdout.raw = raw_dout[i];
        if constexpr (HasDresidualOut) {
          sdres.raw = ld_global_cg_u128(dresidual_out + row * row_stride + col_base);
        }
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
          float const dout_val = to_float_t<T>(sdout.elem[j]);
          float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                         : w0[i].elem[j];
          float const wdy = dout_val * w;
          float grad = fma_ptx(-x_hat, mean_rstd, wdy * rstd);
          if constexpr (HasDresidualOut) {
            grad += to_float_t<T>(sdres.elem[j]);
          }
          sdx.elem[j] = from_float_t<T>(grad);
          dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
        }
        st_global_cs_u128(dx + row * row_stride + col_base, sdx.raw);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
    if constexpr (kVec == 8) {
      st_global_v4_f32(dw_partial + partial_id * row_stride + col_base,
                       dw_accum[i * kVec + 0],
                       dw_accum[i * kVec + 1],
                       dw_accum[i * kVec + 2],
                       dw_accum[i * kVec + 3]);
      st_global_v4_f32(dw_partial + partial_id * row_stride + col_base + 4,
                       dw_accum[i * kVec + 4],
                       dw_accum[i * kVec + 5],
                       dw_accum[i * kVec + 6],
                       dw_accum[i * kVec + 7]);
    } else {
      st_global_v4_f32(dw_partial + partial_id * row_stride + col_base,
                       dw_accum[i * kVec + 0],
                       dw_accum[i * kVec + 1],
                       dw_accum[i * kVec + 2],
                       dw_accum[i * kVec + 3]);
    }
  }
}

__global__ void rmsnorm_reduce_dw_partial_kernel(
    float const* __restrict__ dw_partial,
    float* __restrict__ dw,
    int64_t partial_count,
    int64_t N) {
  extern __shared__ float workspace[];
  int64_t const col = int64_t(blockIdx.x);
  float thread_sum = 0.0f;
  for (int64_t p = threadIdx.x; p < partial_count; p += blockDim.x) {
    thread_sum += dw_partial[p * N + col];
  }
  float const total = block_reduce_sum(thread_sum, workspace);
  if (threadIdx.x == 0) {
    dw[col] = total;
  }
}

__global__ void rmsnorm_reduce_dw_partial_tile32_kernel(
    float const* __restrict__ dw_partial,
    float* __restrict__ dw,
    int64_t partial_count,
    int64_t N) {
  extern __shared__ float workspace[];
  int const lane_col = threadIdx.x & 31;
  int const lane_group = threadIdx.x >> 5;
  int64_t const col = int64_t(blockIdx.x) * 32 + lane_col;

  float thread_sum = 0.0f;
  if (col < N) {
    int64_t p = lane_group;
    for (; p + 24 < partial_count; p += 32) {
      thread_sum += dw_partial[p * N + col];
      thread_sum += dw_partial[(p + 8) * N + col];
      thread_sum += dw_partial[(p + 16) * N + col];
      thread_sum += dw_partial[(p + 24) * N + col];
    }
    for (; p < partial_count; p += 8) {
      thread_sum += dw_partial[p * N + col];
    }
  }
  workspace[lane_group * 32 + lane_col] = thread_sum;
  __syncthreads();

  if (lane_group == 0 && col < N) {
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      total += workspace[i * 32 + lane_col];
    }
    dw[col] = total;
  }
}

__global__ void rmsnorm_reduce_dw_partial_tile16_kernel(
    float const* __restrict__ dw_partial,
    float* __restrict__ dw,
    int64_t partial_count,
    int64_t N) {
  extern __shared__ float workspace[];
  int const lane_col = threadIdx.x & 15;
  int const lane_group = threadIdx.x >> 4;
  int64_t const col = int64_t(blockIdx.x) * 16 + lane_col;

  float thread_sum = 0.0f;
  if (col < N) {
    int64_t p = lane_group;
    for (; p + 48 < partial_count; p += 64) {
      thread_sum += dw_partial[p * N + col];
      thread_sum += dw_partial[(p + 16) * N + col];
      thread_sum += dw_partial[(p + 32) * N + col];
      thread_sum += dw_partial[(p + 48) * N + col];
    }
    for (; p < partial_count; p += 16) {
      thread_sum += dw_partial[p * N + col];
    }
  }
  workspace[lane_group * 16 + lane_col] = thread_sum;
  __syncthreads();

  if (lane_group == 0 && col < N) {
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      total += workspace[i * 16 + lane_col];
    }
    dw[col] = total;
  }
}

template <typename T,
          int ThreadsPerRow,
          int MaxVecs,
          bool AccumulateDw,
          bool PrefetchNextRow = false>
__global__ void rmsnorm_bwd_split_rowdot_kernel(
    T const* __restrict__ x,
    float const* __restrict__ weight,
    T const* __restrict__ dout,
    float const* __restrict__ rstd_in,
    float* __restrict__ row_dot_partial,
    float* __restrict__ dw_partial,
    int64_t M,
    int64_t N,
    int64_t slices,
    int64_t segment_cols) {
  constexpr int kVec = Vec128<T>::kElements;
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const lane = threadIdx.x;
  int64_t const partial_id = int64_t(blockIdx.x);
  int64_t const partial_count = int64_t(gridDim.x);
  int64_t const slice = int64_t(blockIdx.y);
  int64_t const segment_base = slice * segment_cols;

  Float4Raw w0[MaxVecs];
  Float4Raw w1[MaxVecs];
  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
    if constexpr (sizeof(T) == 2) {
      w0[i].raw = ld_global_ca_u128(weight + col_base);
      w1[i].raw = ld_global_ca_u128(weight + col_base + 4);
    } else {
      w0[i].raw = ld_global_ca_u128(weight + col_base);
    }
  }

  float dw_accum[AccumulateDw ? MaxVecs * kVec : 1];
  if constexpr (AccumulateDw) {
    #pragma unroll
    for (int i = 0; i < MaxVecs * kVec; ++i) {
      dw_accum[i] = 0.0f;
    }
  }

  if constexpr (PrefetchNextRow) {
    int64_t row = partial_id;
    bool valid_row = row < M;
    uint4 raw_x_cache[MaxVecs];
    uint4 raw_dout_cache[MaxVecs];
    float rstd = 0.0f;
    if (valid_row) {
      rstd = load_warp_uniform(rstd_in, row);
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
        raw_x_cache[i] = ld_global_cg_u128(x + row * N + col_base);
        raw_dout_cache[i] = ld_global_cg_u128(dout + row * N + col_base);
      }
    }

    while (valid_row) {
      float thread_dot = 0.0f;
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        typename Vec128<T>::Storage sx;
        typename Vec128<T>::Storage sdout;
        sx.raw = raw_x_cache[i];
        sdout.raw = raw_dout_cache[i];
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
          float const dout_val = to_float_t<T>(sdout.elem[j]);
          float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                         : w0[i].elem[j];
          thread_dot = fma_ptx(x_hat, dout_val * w, thread_dot);
          if constexpr (AccumulateDw) {
            dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
          }
        }
      }

      uint4 next_x[MaxVecs];
      uint4 next_dout[MaxVecs];
      int64_t const next_row = row + partial_count;
      bool const has_next = next_row < M;
      float next_rstd = 0.0f;
      if (has_next) {
        next_rstd = load_warp_uniform(rstd_in, next_row);
        #pragma unroll
        for (int i = 0; i < MaxVecs; ++i) {
          int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
          next_x[i] = ld_global_cg_u128(x + next_row * N + col_base);
          next_dout[i] = ld_global_cg_u128(dout + next_row * N + col_base);
        }
      }

      float const slice_dot = row_reduce_sum_lane0<ThreadsPerRow>(thread_dot, workspace);
      if (lane == 0) {
        row_dot_partial[row * slices + slice] = slice_dot;
      }

      if (has_next) {
        #pragma unroll
        for (int i = 0; i < MaxVecs; ++i) {
          raw_x_cache[i] = next_x[i];
          raw_dout_cache[i] = next_dout[i];
        }
      }
      row = next_row;
      rstd = next_rstd;
      valid_row = has_next;
    }
  } else {
    for (int64_t row = partial_id; row < M; row += partial_count) {
      float const rstd = load_warp_uniform(rstd_in, row);
      float thread_dot = 0.0f;
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
        uint4 const raw_x = ld_global_u128(x + row * N + col_base);
        uint4 const raw_dout = ld_global_u128(dout + row * N + col_base);
        typename Vec128<T>::Storage sx;
        typename Vec128<T>::Storage sdout;
        sx.raw = raw_x;
        sdout.raw = raw_dout;
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
          float const dout_val = to_float_t<T>(sdout.elem[j]);
          float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                         : w0[i].elem[j];
          thread_dot = fma_ptx(x_hat, dout_val * w, thread_dot);
          if constexpr (AccumulateDw) {
            dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
          }
        }
      }
      float const slice_dot = row_reduce_sum_lane0<ThreadsPerRow>(thread_dot, workspace);
      if (lane == 0) {
        row_dot_partial[row * slices + slice] = slice_dot;
      }
    }
  }

  if constexpr (AccumulateDw) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
      if constexpr (kVec == 8) {
        st_global_v4_f32(dw_partial + partial_id * N + col_base,
                         dw_accum[i * kVec + 0], dw_accum[i * kVec + 1],
                         dw_accum[i * kVec + 2], dw_accum[i * kVec + 3]);
        st_global_v4_f32(dw_partial + partial_id * N + col_base + 4,
                         dw_accum[i * kVec + 4], dw_accum[i * kVec + 5],
                         dw_accum[i * kVec + 6], dw_accum[i * kVec + 7]);
      } else {
        st_global_v4_f32(dw_partial + partial_id * N + col_base,
                         dw_accum[i * kVec + 0], dw_accum[i * kVec + 1],
                         dw_accum[i * kVec + 2], dw_accum[i * kVec + 3]);
      }
    }
  }
}

template <typename T,
          int ThreadsPerRow,
          int MaxVecs,
          bool AccumulateDw,
          bool HasDresidualOut = false,
          bool PrefetchNextRow = false>
__global__ void rmsnorm_bwd_split_partial_kernel(
    T const* __restrict__ x,
    float const* __restrict__ weight,
    T const* __restrict__ dout,
    float const* __restrict__ rstd_in,
    float const* __restrict__ row_dot_partial,
    float const* __restrict__ row_dot_total,
    T* __restrict__ dx,
    T const* __restrict__ dresidual_out,
    float* __restrict__ dw_partial,
    int64_t M,
    int64_t N,
    int64_t slices,
    int64_t segment_cols) {
  constexpr int kVec = Vec128<T>::kElements;
  int const lane = threadIdx.x;
  int64_t const partial_id = int64_t(blockIdx.x);
  int64_t const partial_count = int64_t(gridDim.x);
  int64_t const slice = int64_t(blockIdx.y);
  int64_t const segment_base = slice * segment_cols;

  Float4Raw w0[MaxVecs];
  Float4Raw w1[MaxVecs];
  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
    if constexpr (sizeof(T) == 2) {
      w0[i].raw = ld_global_ca_u128(weight + col_base);
      w1[i].raw = ld_global_ca_u128(weight + col_base + 4);
    } else {
      w0[i].raw = ld_global_ca_u128(weight + col_base);
    }
  }

  float dw_accum[AccumulateDw ? MaxVecs * kVec : 1];
  if constexpr (AccumulateDw) {
    #pragma unroll
    for (int i = 0; i < MaxVecs * kVec; ++i) {
      dw_accum[i] = 0.0f;
    }
  }

  if constexpr (PrefetchNextRow && !AccumulateDw && !HasDresidualOut) {
    int64_t row = partial_id;
    bool valid_row = row < M;
    uint4 raw_x_cache[MaxVecs];
    uint4 raw_dout_cache[MaxVecs];
    if (valid_row) {
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
        raw_x_cache[i] = ld_global_cg_u128(x + row * N + col_base);
        raw_dout_cache[i] = ld_global_cg_u128(dout + row * N + col_base);
      }
    }
    while (valid_row) {
      float row_dot;
      if (row_dot_total != nullptr) {
        row_dot = load_warp_uniform(row_dot_total, row);
      } else {
        row_dot = 0.0f;
        for (int64_t s = 0; s < slices; ++s) {
          row_dot += row_dot_partial[row * slices + s];
        }
      }
      float const rstd = load_warp_uniform(rstd_in, row);
      float const x_scale = row_dot * (1.0f / static_cast<float>(N)) * rstd * rstd;

      uint4 next_x[MaxVecs];
      uint4 next_dout[MaxVecs];
      int64_t const next_row = row + partial_count;
      bool const has_next = next_row < M;
      if (has_next) {
        #pragma unroll
        for (int i = 0; i < MaxVecs; ++i) {
          int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
          next_x[i] = ld_global_cg_u128(x + next_row * N + col_base);
          next_dout[i] = ld_global_cg_u128(dout + next_row * N + col_base);
        }
      }

      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
        typename Vec128<T>::Storage sx;
        typename Vec128<T>::Storage sdout;
        typename Vec128<T>::Storage sdx;
        sx.raw = raw_x_cache[i];
        sdout.raw = raw_dout_cache[i];
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const x_val = to_float_t<T>(sx.elem[j]);
          float const dout_val = to_float_t<T>(sdout.elem[j]);
          float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                         : w0[i].elem[j];
          sdx.elem[j] = from_float_t<T>(fma_ptx(-x_val, x_scale, dout_val * w * rstd));
        }
        st_global_u128(dx + row * N + col_base, sdx.raw);
      }
      if (has_next) {
        #pragma unroll
        for (int i = 0; i < MaxVecs; ++i) {
          raw_x_cache[i] = next_x[i];
          raw_dout_cache[i] = next_dout[i];
        }
      }
      row = next_row;
      valid_row = has_next;
    }
  } else {
  for (int64_t row = partial_id; row < M; row += partial_count) {
    float row_dot;
    if (row_dot_total != nullptr) {
      row_dot = row_dot_total[row];
    } else {
      row_dot = 0.0f;
      for (int64_t s = 0; s < slices; ++s) {
        row_dot += row_dot_partial[row * slices + s];
      }
    }
    float const rstd = load_warp_uniform(rstd_in, row);
    float const mean = row_dot * (1.0f / static_cast<float>(N));
    float const mean_rstd = mean * rstd;
    float const x_scale = mean_rstd * rstd;

    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
      uint4 const raw_x = ld_global_cg_u128(x + row * N + col_base);
      uint4 const raw_dout = ld_global_cg_u128(dout + row * N + col_base);
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sdout;
      typename Vec128<T>::Storage sdx;
      typename Vec128<T>::Storage sdres;
      sx.raw = raw_x;
      sdout.raw = raw_dout;
      if constexpr (HasDresidualOut) {
        sdres.raw = ld_global_cg_u128(dresidual_out + row * N + col_base);
      }
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_val = to_float_t<T>(sx.elem[j]);
        float const dout_val = to_float_t<T>(sdout.elem[j]);
        float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                       : w0[i].elem[j];
        float const wdy = dout_val * w;
        if constexpr (AccumulateDw) {
          float const x_hat = x_val * rstd;
          float grad = fma_ptx(-x_hat, mean_rstd, wdy * rstd);
          if constexpr (HasDresidualOut) {
            grad += to_float_t<T>(sdres.elem[j]);
          }
          sdx.elem[j] = from_float_t<T>(grad);
          dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
        } else {
          float grad = fma_ptx(-x_val, x_scale, wdy * rstd);
          if constexpr (HasDresidualOut) {
            grad += to_float_t<T>(sdres.elem[j]);
          }
          sdx.elem[j] = from_float_t<T>(grad);
        }
      }
      st_global_u128(dx + row * N + col_base, sdx.raw);
    }
  }
  }

  if constexpr (AccumulateDw) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
      if constexpr (kVec == 8) {
        st_global_v4_f32(dw_partial + partial_id * N + col_base,
                         dw_accum[i * kVec + 0], dw_accum[i * kVec + 1],
                         dw_accum[i * kVec + 2], dw_accum[i * kVec + 3]);
        st_global_v4_f32(dw_partial + partial_id * N + col_base + 4,
                         dw_accum[i * kVec + 4], dw_accum[i * kVec + 5],
                         dw_accum[i * kVec + 6], dw_accum[i * kVec + 7]);
      } else {
        st_global_v4_f32(dw_partial + partial_id * N + col_base,
                         dw_accum[i * kVec + 0], dw_accum[i * kVec + 1],
                         dw_accum[i * kVec + 2], dw_accum[i * kVec + 3]);
      }
    }
  }
}

template <typename T,
          int ThreadsPerRow,
          int MaxVecs,
          int StaticN,
          int DwVecs = 0,
          int SharedDwVecs = 0,
          bool HasDresidualOut = false,
          int ClusterN = 1>
__global__ __launch_bounds__(ThreadsPerRow, 1)
void rmsnorm_bwd_fullrow_dx_only_kernel(
    T const* __restrict__ x,
    float const* __restrict__ weight,
    T const* __restrict__ dout,
    float const* __restrict__ rstd_in,
    T* __restrict__ dx,
    T const* __restrict__ dresidual_out,
    float* __restrict__ dw_partial,
    int64_t M,
    int64_t N) {
  static_assert(sizeof(T) == 2);
  static_assert(DwVecs >= 0 && DwVecs <= MaxVecs);
  static_assert(SharedDwVecs >= 0 && DwVecs + SharedDwVecs <= MaxVecs);
  constexpr int kVec = Vec128<T>::kElements;
  constexpr int kTileVecs = ThreadsPerRow * MaxVecs;
  static_assert(int64_t(kTileVecs) * kVec * ClusterN == StaticN);
  constexpr int64_t kSegmentCols = int64_t(kTileVecs) * kVec;
  constexpr int64_t kRowStride = int64_t(StaticN);
  constexpr float kInvN = 1.0f / static_cast<float>(StaticN);
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  uint4* x_smem = align_t_ptr<uint4>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(x_smem + kTileVecs);
  uint4* dout_smem = align_t_ptr<uint4>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(dout_smem + kTileVecs);
  float* workspace = align_t_ptr<float>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(workspace + 32);
  float* dw_smem = align_t_ptr<float>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(dw_smem + SharedDwVecs * kVec * ThreadsPerRow);
  float* cluster_sums = align_t_ptr<float>(smem_ptr);

  cg::cluster_group cluster = cg::this_cluster();
  int const rank = ClusterN > 1 ? int(cluster.block_rank()) : 0;
  int const lane = threadIdx.x;
  int64_t const partial_id = int64_t(blockIdx.x) / ClusterN;
  int64_t const partial_count = int64_t(gridDim.x) / ClusterN;
  int64_t const segment_base = int64_t(rank) * kSegmentCols;

  Float4Raw w0[MaxVecs];
  Float4Raw w1[MaxVecs];
  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
    w0[i].raw = ld_global_u128(weight + col_base);
    w1[i].raw = ld_global_u128(weight + col_base + 4);
  }

  float dw_accum[DwVecs > 0 ? DwVecs * kVec : 1];
  if constexpr (DwVecs > 0) {
    #pragma unroll
    for (int i = 0; i < DwVecs * kVec; ++i) {
      dw_accum[i] = 0.0f;
    }
  }
  if constexpr (SharedDwVecs > 0) {
    #pragma unroll
    for (int i = 0; i < SharedDwVecs * kVec; ++i) {
      dw_smem[i * ThreadsPerRow + lane] = 0.0f;
    }
  }

  for (int64_t row = partial_id; row < M; row += partial_count) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int const vec = lane + i * ThreadsPerRow;
      int64_t const col_base = segment_base + int64_t(vec) * kVec;
      cp_async_ca_shared_global_16(x_smem + vec, x + row * kRowStride + col_base);
      cp_async_ca_shared_global_16(dout_smem + vec, dout + row * kRowStride + col_base);
      if constexpr (HasDresidualOut) {
        if ((lane & 7) == 0) {
          prefetch_global_l2(dresidual_out + row * kRowStride + col_base);
        }
      }
    }
    cp_async_commit_group();
    cp_async_wait_group<0>();

    float const rstd = rstd_in[row];
    float thread_dot = 0.0f;
    if constexpr (DwVecs > 0) {
      #pragma unroll
      for (int i = 0; i < DwVecs; ++i) {
        int const vec = lane + i * ThreadsPerRow;
        typename Vec128<T>::Storage sx;
        typename Vec128<T>::Storage sdout;
        sx.raw = x_smem[vec];
        sdout.raw = dout_smem[vec];
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
          float const dout_val = to_float_t<T>(sdout.elem[j]);
          float const w = j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4];
          thread_dot = fma_ptx(x_hat, dout_val * w, thread_dot);
          dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
        }
      }
    }
    if constexpr (SharedDwVecs > 0) {
      #pragma unroll
      for (int s = 0; s < SharedDwVecs; ++s) {
        int const i = DwVecs + s;
        int const vec = lane + i * ThreadsPerRow;
        typename Vec128<T>::Storage sx;
        typename Vec128<T>::Storage sdout;
        sx.raw = x_smem[vec];
        sdout.raw = dout_smem[vec];
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
          float const dout_val = to_float_t<T>(sdout.elem[j]);
          float const w = j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4];
          thread_dot = fma_ptx(x_hat, dout_val * w, thread_dot);
          int const idx = s * kVec + j;
          float const accum = dw_smem[idx * ThreadsPerRow + lane];
          dw_smem[idx * ThreadsPerRow + lane] = fma_ptx(dout_val, x_hat, accum);
        }
      }
    }
    #pragma unroll
    for (int i = DwVecs + SharedDwVecs; i < MaxVecs; ++i) {
      int const vec = lane + i * ThreadsPerRow;
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sdout;
      sx.raw = x_smem[vec];
      sdout.raw = dout_smem[vec];
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
        float const dout_val = to_float_t<T>(sdout.elem[j]);
        float const w = j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4];
        thread_dot = fma_ptx(x_hat, dout_val * w, thread_dot);
      }
    }

    float row_dot = row_reduce_sum<ThreadsPerRow>(thread_dot, workspace);
    if constexpr (ClusterN > 1) {
      if (lane == 0) {
        float* rank0_sums = static_cast<float*>(cluster.map_shared_rank(cluster_sums, 0));
        rank0_sums[rank] = row_dot;
      }
      cluster_barrier_inline();
      if (lane == 0) {
        float total = 0.0f;
        float* rank0_sums = static_cast<float*>(cluster.map_shared_rank(cluster_sums, 0));
        #pragma unroll
        for (int i = 0; i < ClusterN; ++i) {
          total += rank0_sums[i];
        }
        workspace[0] = total;
      }
      __syncthreads();
      row_dot = workspace[0];
    }
    float const x_scale = row_dot * kInvN * rstd * rstd;

    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int const vec = lane + i * ThreadsPerRow;
      int64_t const col_base = segment_base + int64_t(vec) * kVec;
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sdout;
      typename Vec128<T>::Storage sdx;
      typename Vec128<T>::Storage sdres;
      sx.raw = x_smem[vec];
      sdout.raw = dout_smem[vec];
      if constexpr (HasDresidualOut) {
        sdres.raw = ld_global_cg_u128(dresidual_out + row * kRowStride + col_base);
      }
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_val = to_float_t<T>(sx.elem[j]);
        float const dout_val = to_float_t<T>(sdout.elem[j]);
        float const w = j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4];
        float const wdy = dout_val * w;
        float grad = fma_ptx(-x_val, x_scale, wdy * rstd);
        if constexpr (HasDresidualOut) {
          grad += to_float_t<T>(sdres.elem[j]);
        }
        sdx.elem[j] = from_float_t<T>(grad);
      }
      st_global_u128(dx + row * kRowStride + col_base, sdx.raw);
    }
    if constexpr (ClusterN > 1) {
      cluster_barrier_inline_relaxed();
    }
  }

  if constexpr (DwVecs > 0) {
    #pragma unroll
    for (int i = 0; i < DwVecs; ++i) {
      int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
      st_global_v4_f32(dw_partial + partial_id * kRowStride + col_base,
                       dw_accum[i * kVec + 0],
                       dw_accum[i * kVec + 1],
                       dw_accum[i * kVec + 2],
                       dw_accum[i * kVec + 3]);
      st_global_v4_f32(dw_partial + partial_id * kRowStride + col_base + 4,
                       dw_accum[i * kVec + 4],
                       dw_accum[i * kVec + 5],
                       dw_accum[i * kVec + 6],
                       dw_accum[i * kVec + 7]);
    }
  }
  if constexpr (SharedDwVecs > 0) {
    #pragma unroll
    for (int s = 0; s < SharedDwVecs; ++s) {
      int const i = DwVecs + s;
      int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
      int const idx = s * kVec;
      st_global_v4_f32(dw_partial + partial_id * kRowStride + col_base,
                       dw_smem[(idx + 0) * ThreadsPerRow + lane],
                       dw_smem[(idx + 1) * ThreadsPerRow + lane],
                       dw_smem[(idx + 2) * ThreadsPerRow + lane],
                       dw_smem[(idx + 3) * ThreadsPerRow + lane]);
      st_global_v4_f32(dw_partial + partial_id * kRowStride + col_base + 4,
                       dw_smem[(idx + 4) * ThreadsPerRow + lane],
                       dw_smem[(idx + 5) * ThreadsPerRow + lane],
                       dw_smem[(idx + 6) * ThreadsPerRow + lane],
                       dw_smem[(idx + 7) * ThreadsPerRow + lane]);
    }
  }
}

template <typename T, int ThreadsPerRow, int MaxVecs>
__global__ void rmsnorm_bwd_split_dw_only_kernel(
    T const* __restrict__ x,
    T const* __restrict__ dout,
    float const* __restrict__ rstd_in,
    float* __restrict__ dw_partial,
    int64_t M,
    int64_t N,
    int64_t segment_cols,
    int64_t slice_offset) {
  static_assert(sizeof(T) == 2);
  constexpr int kVec = Vec128<T>::kElements;
  int const lane = threadIdx.x;
  int64_t const partial_id = int64_t(blockIdx.x);
  int64_t const partial_count = int64_t(gridDim.x);
  int64_t const slice = int64_t(blockIdx.y) + slice_offset;
  int64_t const segment_base = slice * segment_cols;

  float dw_accum[MaxVecs * kVec];
  #pragma unroll
  for (int i = 0; i < MaxVecs * kVec; ++i) {
    dw_accum[i] = 0.0f;
  }

  for (int64_t row = partial_id; row < M; row += partial_count) {
    float const rstd = load_warp_uniform(rstd_in, row);
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
      uint4 const raw_x = ld_global_u128(x + row * N + col_base);
      uint4 const raw_dout = ld_global_u128(dout + row * N + col_base);
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sdout;
      sx.raw = raw_x;
      sdout.raw = raw_dout;
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
        float const dout_val = to_float_t<T>(sdout.elem[j]);
        dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = segment_base + int64_t(lane + i * ThreadsPerRow) * kVec;
    st_global_v4_f32(dw_partial + partial_id * N + col_base,
                     dw_accum[i * kVec + 0], dw_accum[i * kVec + 1],
                     dw_accum[i * kVec + 2], dw_accum[i * kVec + 3]);
    st_global_v4_f32(dw_partial + partial_id * N + col_base + 4,
                     dw_accum[i * kVec + 4], dw_accum[i * kVec + 5],
                     dw_accum[i * kVec + 6], dw_accum[i * kVec + 7]);
  }
}

__global__ void rmsnorm_reduce_rowdot_partial_kernel(
    float const* __restrict__ row_dot_partial,
    float* __restrict__ row_dot_total,
    int64_t M,
    int64_t slices) {
  int64_t const row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= M) {
    return;
  }
  float total = 0.0f;
  for (int64_t s = 0; s < slices; ++s) {
    total += row_dot_partial[row * slices + s];
  }
  row_dot_total[row] = total;
}

template <int ThreadsPerRow, int MaxVecs>
__global__ void rmsnorm_bwd_contig_fp32_stream_kernel(
    float const* __restrict__ x,
    float const* __restrict__ weight,
    float const* __restrict__ dout,
    float const* __restrict__ rstd_in,
    float* __restrict__ dx,
    float* __restrict__ dw_partial,
    int64_t M,
    int64_t N) {
  constexpr int kVec = Vec128<float>::kElements;
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const lane = threadIdx.x;
  int64_t const partial_id = int64_t(blockIdx.x);
  int64_t const partial_count = int64_t(gridDim.x);

  float dw_accum[MaxVecs * kVec];
  #pragma unroll
  for (int i = 0; i < MaxVecs * kVec; ++i) {
    dw_accum[i] = 0.0f;
  }

  for (int64_t row = partial_id; row < M; row += partial_count) {
    float const rstd = rstd_in[row];
    float thread_dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
      uint4 const raw_x = ld_global_u128(x + row * N + col_base);
      uint4 const raw_dout = ld_global_u128(dout + row * N + col_base);
      uint4 const raw_w = ld_global_u128(weight + col_base);
      typename Vec128<float>::Storage sx;
      typename Vec128<float>::Storage sdout;
      Float4Raw sw;
      sx.raw = raw_x;
      sdout.raw = raw_dout;
      sw.raw = raw_w;
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_hat = sx.elem[j] * rstd;
        float const dout_val = sdout.elem[j];
        thread_dot = fma_ptx(x_hat, dout_val * sw.elem[j], thread_dot);
        dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
      }
    }

    float const row_dot = row_reduce_sum<ThreadsPerRow>(thread_dot, workspace);
    float const mean = row_dot * (1.0f / static_cast<float>(N));
    float const mean_rstd = mean * rstd;

    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
      uint4 const raw_x = ld_global_u128(x + row * N + col_base);
      uint4 const raw_dout = ld_global_u128(dout + row * N + col_base);
      uint4 const raw_w = ld_global_u128(weight + col_base);
      typename Vec128<float>::Storage sx;
      typename Vec128<float>::Storage sdout;
      typename Vec128<float>::Storage sdx;
      Float4Raw sw;
      sx.raw = raw_x;
      sdout.raw = raw_dout;
      sw.raw = raw_w;
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_hat = sx.elem[j] * rstd;
        float const wdy = sdout.elem[j] * sw.elem[j];
        sdx.elem[j] = fma_ptx(-x_hat, mean_rstd, wdy * rstd);
      }
      st_global_u128(dx + row * N + col_base, sdx.raw);
    }
  }

  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
    #pragma unroll
    for (int j = 0; j < kVec; ++j) {
      dw_partial[partial_id * N + col_base + j] = dw_accum[i * kVec + j];
    }
  }
}

template <int ThreadsPerRow, int MaxVecs>
__global__ void rmsnorm_bwd_contig_fp32_smem_dw_kernel(
    float const* __restrict__ x,
    float const* __restrict__ weight,
    float const* __restrict__ dout,
    float const* __restrict__ rstd_in,
    float* __restrict__ dx,
    float* __restrict__ dw_partial,
    int64_t M,
    int64_t N) {
  constexpr int kVec = Vec128<float>::kElements;
  constexpr int kDwValues = MaxVecs * kVec;
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(workspace + 32);
  float* dw_smem = align_t_ptr<float>(smem_ptr);

  int const lane = threadIdx.x;
  int64_t const partial_id = int64_t(blockIdx.x);
  int64_t const partial_count = int64_t(gridDim.x);

  Float4Raw sw[MaxVecs];
  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
    sw[i].raw = ld_global_u128(weight + col_base);
  }

  #pragma unroll
  for (int idx = 0; idx < kDwValues; ++idx) {
    dw_smem[idx * ThreadsPerRow + lane] = 0.0f;
  }

  for (int64_t row = partial_id; row < M; row += partial_count) {
    uint4 raw_x[MaxVecs];
    uint4 raw_dout[MaxVecs];
    float const rstd = rstd_in[row];
    float thread_dot = 0.0f;

    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
      raw_x[i] = ld_global_u128(x + row * N + col_base);
      raw_dout[i] = ld_global_u128(dout + row * N + col_base);
      typename Vec128<float>::Storage sx;
      typename Vec128<float>::Storage sdout;
      sx.raw = raw_x[i];
      sdout.raw = raw_dout[i];
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_hat = sx.elem[j] * rstd;
        float const dout_val = sdout.elem[j];
        thread_dot = fma_ptx(x_hat, dout_val * sw[i].elem[j], thread_dot);
      }
    }

    float const row_dot = row_reduce_sum<ThreadsPerRow>(thread_dot, workspace);
    float const mean = row_dot * (1.0f / static_cast<float>(N));
    float const mean_rstd = mean * rstd;

    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
      typename Vec128<float>::Storage sx;
      typename Vec128<float>::Storage sdout;
      typename Vec128<float>::Storage sdx;
      sx.raw = raw_x[i];
      sdout.raw = raw_dout[i];
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_hat = sx.elem[j] * rstd;
        float const dout_val = sdout.elem[j];
        float const wdy = dout_val * sw[i].elem[j];
        sdx.elem[j] = fma_ptx(-x_hat, mean_rstd, wdy * rstd);
        int const idx = i * kVec + j;
        float const accum = dw_smem[idx * ThreadsPerRow + lane];
        dw_smem[idx * ThreadsPerRow + lane] = fma_ptx(dout_val, x_hat, accum);
      }
      st_global_u128(dx + row * N + col_base, sdx.raw);
    }
  }

  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
    #pragma unroll
    for (int j = 0; j < kVec; ++j) {
      int const idx = i * kVec + j;
      dw_partial[partial_id * N + col_base + j] = dw_smem[idx * ThreadsPerRow + lane];
    }
  }
}

template <typename T,
          int ThreadsPerRow,
          int MaxVecs,
          int StaticN = 0,
          bool HasDresidualOut = false>
__global__ void rmsnorm_bwd_contig_cp_async_kernel(
    T const* __restrict__ x,
    float const* __restrict__ weight,
    T const* __restrict__ dout,
    float const* __restrict__ rstd_in,
    T* __restrict__ dx,
    T const* __restrict__ dresidual_out,
    float* __restrict__ dw_partial,
    int64_t M,
    int64_t N) {
  constexpr int kVec = Vec128<T>::kElements;
  constexpr int kTileVecs = ThreadsPerRow * MaxVecs;
  constexpr bool kStaticN = StaticN > 0;
  constexpr bool kEarlyIssueNextRow = StaticN == 4096 || StaticN == 16384 ||
                                      (StaticN == 8192 && sizeof(T) == 4);
  int64_t const row_stride = kStaticN ? int64_t(StaticN) : N;
  float const inv_n = kStaticN ? (1.0f / static_cast<float>(StaticN))
                               : (1.0f / static_cast<float>(N));
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  uint4* x_smem = align_t_ptr<uint4>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(x_smem + 2 * kTileVecs);
  uint4* dout_smem = align_t_ptr<uint4>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(dout_smem + 2 * kTileVecs);
  uint4* dres_smem = nullptr;
  if constexpr (HasDresidualOut) {
    dres_smem = align_t_ptr<uint4>(smem_ptr);
    smem_ptr = reinterpret_cast<unsigned char*>(dres_smem + 2 * kTileVecs);
  }
  float* workspace = align_t_ptr<float>(smem_ptr);

  int const lane = threadIdx.x;
  int64_t const partial_id = int64_t(blockIdx.x);
  int64_t const partial_count = int64_t(gridDim.x);

  Float4Raw w0[MaxVecs];
  Float4Raw w1[sizeof(T) == 2 ? MaxVecs : 1];
  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
    w0[i].raw = ld_global_ca_u128(weight + col_base);
    if constexpr (sizeof(T) == 2) {
      w1[i].raw = ld_global_ca_u128(weight + col_base + 4);
    }
  }

  float dw_accum[MaxVecs * kVec];
  #pragma unroll
  for (int i = 0; i < MaxVecs * kVec; ++i) {
    dw_accum[i] = 0.0f;
  }

  auto issue_row = [&](int64_t row, int stage) {
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int const vec = lane + i * ThreadsPerRow;
      int64_t const col_base = int64_t(vec) * kVec;
      cp_async_ca_shared_global_16(
          x_smem + stage * kTileVecs + vec, x + row * row_stride + col_base);
      cp_async_ca_shared_global_16(
          dout_smem + stage * kTileVecs + vec, dout + row * row_stride + col_base);
      if constexpr (HasDresidualOut) {
        cp_async_ca_shared_global_16(dres_smem + stage * kTileVecs + vec,
                                     dresidual_out + row * row_stride + col_base);
      }
    }
    cp_async_commit_group();
  };

  int64_t row = partial_id;
  bool valid_row = row < M;
  int stage = 0;
  if (valid_row) {
    issue_row(row, stage);
  }

  while (valid_row) {
    cp_async_wait_group<0>();

    uint4 raw_x[MaxVecs];
    float const rstd = rstd_in[row];
    float thread_dot = 0.0f;
    int64_t const next_row = row + partial_count;
    bool const has_next = next_row < M;
    int const next_stage = stage ^ 1;
    if constexpr (kEarlyIssueNextRow) {
      if (has_next) {
        issue_row(next_row, next_stage);
      }
    }

    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int const vec = lane + i * ThreadsPerRow;
      raw_x[i] = x_smem[stage * kTileVecs + vec];
      uint4 const raw_dout = dout_smem[stage * kTileVecs + vec];
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sdout;
      sx.raw = raw_x[i];
      sdout.raw = raw_dout;
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_hat = to_float_t<T>(sx.elem[j]) * rstd;
        float const dout_val = to_float_t<T>(sdout.elem[j]);
        float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                       : w0[i].elem[j];
        thread_dot = fma_ptx(x_hat, dout_val * w, thread_dot);
        dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
      }
    }

    if constexpr (!kEarlyIssueNextRow) {
      if (has_next) {
        issue_row(next_row, next_stage);
      }
    }

    float const row_dot = row_reduce_sum<ThreadsPerRow>(thread_dot, workspace);
    float const mean = row_dot * inv_n;
    float const x_scale = mean * rstd * rstd;

    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int const vec = lane + i * ThreadsPerRow;
      int64_t const col_base = int64_t(vec) * kVec;
      typename Vec128<T>::Storage sx;
      typename Vec128<T>::Storage sdout;
      typename Vec128<T>::Storage sdx;
      typename Vec128<T>::Storage sdres;
      sx.raw = raw_x[i];
      sdout.raw = dout_smem[stage * kTileVecs + vec];
      if constexpr (HasDresidualOut) {
        sdres.raw = dres_smem[stage * kTileVecs + vec];
      }
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_val = to_float_t<T>(sx.elem[j]);
        float const dout_val = to_float_t<T>(sdout.elem[j]);
        float const w = sizeof(T) == 2 ? (j < 4 ? w0[i].elem[j] : w1[i].elem[j - 4])
                                       : w0[i].elem[j];
        float const wdy = dout_val * w;
        float grad = fma_ptx(-x_val, x_scale, wdy * rstd);
        if constexpr (HasDresidualOut) {
          grad += to_float_t<T>(sdres.elem[j]);
        }
        sdx.elem[j] = from_float_t<T>(grad);
      }
      st_global_u128(dx + row * row_stride + col_base, sdx.raw);
    }

    row = next_row;
    valid_row = has_next;
    stage = next_stage;
  }

  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = int64_t(lane + i * ThreadsPerRow) * kVec;
    if constexpr (kVec == 8) {
      st_global_v4_f32(dw_partial + partial_id * row_stride + col_base,
                       dw_accum[i * kVec + 0],
                       dw_accum[i * kVec + 1],
                       dw_accum[i * kVec + 2],
                       dw_accum[i * kVec + 3]);
      st_global_v4_f32(dw_partial + partial_id * row_stride + col_base + 4,
                       dw_accum[i * kVec + 4],
                       dw_accum[i * kVec + 5],
                       dw_accum[i * kVec + 6],
                       dw_accum[i * kVec + 7]);
    } else {
      if constexpr (kVec == 4) {
        st_global_v4_f32(dw_partial + partial_id * row_stride + col_base,
                         dw_accum[i * kVec + 0],
                         dw_accum[i * kVec + 1],
                         dw_accum[i * kVec + 2],
                         dw_accum[i * kVec + 3]);
      } else {
        #pragma unroll
        for (int j = 0; j < kVec; ++j) {
          dw_partial[partial_id * row_stride + col_base + j] = dw_accum[i * kVec + j];
        }
      }
    }
  }
}


template <int ClusterN,
          int ThreadsPerRow,
          int MaxVecs,
          bool DistributedSums = false,
          bool DoubleBufferedSums = false,
          bool StaticN = false,
          bool InlineClusterBarrier = false,
          bool UseRegularLoads = false,
          int MinBlocksPerSm = 1,
          bool HasDresidualOut = false,
          bool StageDresidual = false,
          bool DelayClusterReuseBarrier = false,
          bool UseMbarrierReduce = false>
__global__ __launch_bounds__(ThreadsPerRow, MinBlocksPerSm)
void rmsnorm_bwd_contig_fp32_cluster_reg_kernel(
    float const* __restrict__ x,
    float const* __restrict__ weight,
    float const* __restrict__ dout,
    float const* __restrict__ rstd_in,
    float* __restrict__ dx,
    float const* __restrict__ dresidual_out,
    float* __restrict__ dw_partial,
    int64_t M,
    int64_t N) {
  constexpr int kVec = Vec128<float>::kElements;
  constexpr int64_t kSegmentCols = int64_t(ThreadsPerRow) * kVec * MaxVecs;
  constexpr int64_t kStaticN = int64_t(ClusterN) * kSegmentCols;
  static_assert(!StageDresidual || HasDresidualOut);
  static_assert(!UseMbarrierReduce || DelayClusterReuseBarrier);
  int64_t const row_stride = StaticN ? kStaticN : N;
  float const inv_n = StaticN ? (1.0f / static_cast<float>(kStaticN))
                              : (1.0f / static_cast<float>(N));
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;
  float* workspace = align_t_ptr<float>(smem_ptr);
  smem_ptr = reinterpret_cast<unsigned char*>(workspace + 32);
  float* cluster_sums = align_t_ptr<float>(smem_ptr);
  constexpr int kClusterSumValues =
      UseMbarrierReduce ? (ThreadsPerRow / 32) * ClusterN
                        : (DoubleBufferedSums ? 32 : ClusterN);
  smem_ptr = reinterpret_cast<unsigned char*>(cluster_sums + kClusterSumValues);
  uint64_t* cluster_mbar = nullptr;
  if constexpr (UseMbarrierReduce) {
    cluster_mbar = align_t_ptr<uint64_t>(smem_ptr);
    smem_ptr = reinterpret_cast<unsigned char*>(cluster_mbar + 1);
  }
  uint4* dres_smem = nullptr;
  if constexpr (HasDresidualOut && StageDresidual) {
    dres_smem = align_t_ptr<uint4>(smem_ptr);
    smem_ptr = reinterpret_cast<unsigned char*>(dres_smem + 2 * ThreadsPerRow * MaxVecs);
  }

  cg::cluster_group cluster = cg::this_cluster();
  int const rank = int(cluster.block_rank());
  int const lane = threadIdx.x;
  if constexpr (UseMbarrierReduce) {
    if (lane == 0) {
      mbarrier_init_local(cluster_mbar, 1);
    }
    mbarrier_init_fence_cluster();
    if constexpr (InlineClusterBarrier) {
      cluster_barrier_inline();
    } else {
      cluster.sync();
    }
  }
  int64_t const partial_id = int64_t(blockIdx.x) / ClusterN;
  int64_t const partial_count = int64_t(gridDim.x) / ClusterN;
  int64_t const segment_base = int64_t(rank) * kSegmentCols;
  int64_t const lane_col_base = segment_base + int64_t(lane) * kVec;
  constexpr int64_t kColStride = int64_t(ThreadsPerRow) * kVec;
  constexpr int kTileVecs = ThreadsPerRow * MaxVecs;

  auto issue_dres_row = [&](int64_t load_row, int load_stage) {
    if constexpr (HasDresidualOut && StageDresidual) {
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int const vec = lane + i * ThreadsPerRow;
        int64_t const col_base = lane_col_base + int64_t(i) * kColStride;
        cp_async_ca_shared_global_16(dres_smem + load_stage * kTileVecs + vec,
                                     dresidual_out + load_row * row_stride + col_base);
      }
      cp_async_commit_group();
    }
  };

  Float4Raw sw[MaxVecs];
  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = lane_col_base + int64_t(i) * kColStride;
    sw[i].raw = ld_global_ca_u128(weight + col_base);
  }

  float dw_accum[MaxVecs * kVec];
  #pragma unroll
  for (int i = 0; i < MaxVecs * kVec; ++i) {
    dw_accum[i] = 0.0f;
  }

  int64_t row = partial_id;
  bool valid_row = row < M;
  uint4 raw_x_cache[MaxVecs];
  uint4 raw_dout_cache[MaxVecs];
  uint4 raw_dres_cache[HasDresidualOut ? MaxVecs : 1];
  float rstd = 0.0f;
  if (valid_row) {
    rstd = rstd_in[row];
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = lane_col_base + int64_t(i) * kColStride;
      if constexpr (UseRegularLoads) {
        raw_x_cache[i] = ld_global_u128(x + row * row_stride + col_base);
        raw_dout_cache[i] = ld_global_u128(dout + row * row_stride + col_base);
      } else {
        raw_x_cache[i] = ld_global_cg_u128(x + row * row_stride + col_base);
        raw_dout_cache[i] = ld_global_cg_u128(dout + row * row_stride + col_base);
      }
      if constexpr (HasDresidualOut && !StageDresidual) {
        if constexpr (UseRegularLoads) {
          raw_dres_cache[i] = ld_global_u128(dresidual_out + row * row_stride + col_base);
        } else {
          raw_dres_cache[i] = ld_global_cg_u128(dresidual_out + row * row_stride + col_base);
        }
      }
    }
    issue_dres_row(row, 0);
  }

  int reduce_stage = 0;
  int dres_stage = 0;
  while (valid_row) {
    float thread_dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      typename Vec128<float>::Storage sx;
      typename Vec128<float>::Storage sdout;
      sx.raw = raw_x_cache[i];
      sdout.raw = raw_dout_cache[i];
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_hat = sx.elem[j] * rstd;
        float const dout_val = sdout.elem[j];
        thread_dot = fma_ptx(x_hat, dout_val * sw[i].elem[j], thread_dot);
        dw_accum[i * kVec + j] = fma_ptx(dout_val, x_hat, dw_accum[i * kVec + j]);
      }
    }

    uint4 next_x[MaxVecs];
    uint4 next_dout[MaxVecs];
    uint4 next_dres[HasDresidualOut ? MaxVecs : 1];
    int64_t const next_row = row + partial_count;
    bool const has_next = next_row < M;
    int const next_dres_stage = dres_stage ^ 1;
    float next_rstd = 0.0f;
    if (has_next) {
      next_rstd = rstd_in[next_row];
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        int64_t const col_base = lane_col_base + int64_t(i) * kColStride;
        if constexpr (UseRegularLoads) {
          next_x[i] = ld_global_u128(x + next_row * row_stride + col_base);
          next_dout[i] = ld_global_u128(dout + next_row * row_stride + col_base);
        } else {
          next_x[i] = ld_global_cg_u128(x + next_row * row_stride + col_base);
          next_dout[i] = ld_global_cg_u128(dout + next_row * row_stride + col_base);
        }
        if constexpr (HasDresidualOut && !StageDresidual) {
          if constexpr (UseRegularLoads) {
            next_dres[i] = ld_global_u128(dresidual_out + next_row * row_stride + col_base);
          } else {
            next_dres[i] = ld_global_cg_u128(dresidual_out + next_row * row_stride + col_base);
          }
        }
      }
      issue_dres_row(next_row, next_dres_stage);
    }

    float* cluster_sums_stage =
        cluster_sums + (DoubleBufferedSums ? reduce_stage * 16 : 0);
    if constexpr (UseMbarrierReduce) {
      constexpr int kWarpsPerRow = ThreadsPerRow / 32;
      int const warp = lane >> 5;
      int const lane_in_warp = lane & 31;
      float const warp_dot = warp_reduce_sum_full(thread_dot);
      if (lane == 0) {
        mbarrier_arrive_expect_tx_local(
            cluster_mbar, kWarpsPerRow * ClusterN * int(sizeof(float)));
      }
      if (lane_in_warp < ClusterN) {
        st_async_shared_cluster_mbarrier_f32(
            cluster_sums_stage + warp * ClusterN + rank,
            warp_dot,
            cluster_mbar,
            lane_in_warp);
      }
      mbarrier_wait_parity_local(cluster_mbar, reduce_stage);
      reduce_stage ^= 1;
      if (warp == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < (kWarpsPerRow * ClusterN + 31) / 32; ++i) {
          int const idx = lane_in_warp + i * 32;
          if (idx < kWarpsPerRow * ClusterN) {
            total += cluster_sums_stage[idx];
          }
        }
        total = warp_reduce_sum_full(total);
        if (lane_in_warp == 0) {
          workspace[0] = total;
        }
      }
      __syncthreads();
    } else {
      float const slice_dot = row_reduce_sum<ThreadsPerRow>(thread_dot, workspace);
      if constexpr (DistributedSums) {
        if (lane == 0) {
          cluster_sums_stage[rank] = slice_dot;
        }
        if constexpr (InlineClusterBarrier) {
          cluster_barrier_inline();
        } else {
          cluster.sync();
        }
        if (lane == 0) {
          float total_lane0 = 0.0f;
          #pragma unroll
          for (int i = 0; i < ClusterN; ++i) {
            float* rank_sums = static_cast<float*>(cluster.map_shared_rank(cluster_sums_stage, i));
            total_lane0 += rank_sums[i];
          }
          workspace[0] = total_lane0;
        }
      } else {
        if (lane == 0) {
          float* rank0_sums = static_cast<float*>(cluster.map_shared_rank(cluster_sums_stage, 0));
          rank0_sums[rank] = slice_dot;
        }
        if constexpr (InlineClusterBarrier) {
          cluster_barrier_inline();
        } else {
          cluster.sync();
        }
        if (lane == 0) {
          float total_lane0 = 0.0f;
          float* rank0_sums = static_cast<float*>(cluster.map_shared_rank(cluster_sums_stage, 0));
          #pragma unroll
          for (int i = 0; i < ClusterN; ++i) {
            total_lane0 += rank0_sums[i];
          }
          workspace[0] = total_lane0;
        }
      }
      __syncthreads();
      if constexpr (DoubleBufferedSums) {
        reduce_stage ^= 1;
      } else {
        if constexpr (!DelayClusterReuseBarrier) {
          if constexpr (InlineClusterBarrier) {
            cluster_barrier_inline_relaxed();
          } else {
            cluster.sync();
          }
        }
      }
    }
    float const mean = workspace[0] * inv_n;
    if constexpr (HasDresidualOut && StageDresidual) {
      if (has_next) {
        cp_async_wait_group<1>();
      } else {
        cp_async_wait_group<0>();
      }
    }

    #pragma unroll
    for (int i = 0; i < MaxVecs; ++i) {
      int64_t const col_base = lane_col_base + int64_t(i) * kColStride;
      typename Vec128<float>::Storage sx;
      typename Vec128<float>::Storage sdout;
      Float4Raw sdx;
      Float4Raw sdres;
      sx.raw = raw_x_cache[i];
      sdout.raw = raw_dout_cache[i];
      if constexpr (HasDresidualOut) {
        if constexpr (StageDresidual) {
          int const vec = lane + i * ThreadsPerRow;
          sdres.raw = dres_smem[dres_stage * kTileVecs + vec];
        } else {
          sdres.raw = raw_dres_cache[i];
        }
      }
      #pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const x_hat = sx.elem[j] * rstd;
        float const wdy = sdout.elem[j] * sw[i].elem[j];
        float grad = (wdy - x_hat * mean) * rstd;
        if constexpr (HasDresidualOut) {
          grad += sdres.elem[j];
        }
        sdx.elem[j] = grad;
      }
      st_global_v4_f32(dx + row * row_stride + col_base,
                       sdx.elem[0],
                       sdx.elem[1],
                       sdx.elem[2],
                       sdx.elem[3]);
    }

    if constexpr (!DoubleBufferedSums && DelayClusterReuseBarrier) {
      if constexpr (InlineClusterBarrier) {
        cluster_barrier_inline_relaxed();
      } else {
        cluster.sync();
      }
    }

    if (has_next) {
      #pragma unroll
      for (int i = 0; i < MaxVecs; ++i) {
        raw_x_cache[i] = next_x[i];
        raw_dout_cache[i] = next_dout[i];
        if constexpr (HasDresidualOut && !StageDresidual) {
          raw_dres_cache[i] = next_dres[i];
        }
      }
    }
    row = next_row;
    rstd = next_rstd;
    valid_row = has_next;
    dres_stage = next_dres_stage;
  }

  #pragma unroll
  for (int i = 0; i < MaxVecs; ++i) {
    int64_t const col_base = lane_col_base + int64_t(i) * kColStride;
    st_global_v4_f32(dw_partial + partial_id * row_stride + col_base,
                     dw_accum[i * kVec + 0],
                     dw_accum[i * kVec + 1],
                     dw_accum[i * kVec + 2],
                     dw_accum[i * kVec + 3]);
  }
}


DType dtype_from_tensor(torch::Tensor const& t, bool allow_empty_none = true) {
  if (allow_empty_none && t.numel() == 0) {
    return DType::kNone;
  }
  switch (t.scalar_type()) {
    case at::ScalarType::Half:
      return DType::kHalf;
    case at::ScalarType::BFloat16:
      return DType::kBFloat16;
    case at::ScalarType::Float:
      return DType::kFloat;
    default:
      TORCH_CHECK(false, "unsupported dtype: ", t.scalar_type());
  }
}

Tensor3 make_tensor3(torch::Tensor const& t, bool empty_is_none = true) {
  Tensor3 v{};
  v.ptr = t.numel() == 0 && empty_is_none ? nullptr : t.data_ptr();
  v.dtype = dtype_from_tensor(t, empty_is_none);
  if (v.dtype == DType::kNone) {
    v.stride_m = v.stride_h = v.stride_n = 0;
    return v;
  }
  TORCH_CHECK(t.dim() == 2 || t.dim() == 3, "expected a 2D or 3D tensor");
  TORCH_CHECK(t.stride(t.dim() - 1) == 1, "last dimension must be contiguous");
  if (t.dim() == 2) {
    v.stride_m = t.stride(0);
    v.stride_h = 0;
    v.stride_n = t.stride(1);
  } else {
    v.stride_m = t.stride(0);
    v.stride_h = t.stride(1);
    v.stride_n = t.stride(2);
  }
  return v;
}

Tensor2 make_tensor2(torch::Tensor const& t, int x_dim, bool empty_is_none = true) {
  Tensor2 v{};
  v.ptr = t.numel() == 0 && empty_is_none ? nullptr : t.data_ptr();
  v.dtype = dtype_from_tensor(t, empty_is_none);
  if (v.dtype == DType::kNone) {
    v.stride_m = v.stride_h = 0;
    return v;
  }
  TORCH_CHECK(t.scalar_type() == at::ScalarType::Float, "rstd must be float32");
  TORCH_CHECK(t.dim() == x_dim - 1, "rstd rank must be one less than x rank");
  if (t.dim() == 1) {
    v.stride_m = t.stride(0);
    v.stride_h = 0;
  } else {
    v.stride_m = t.stride(0);
    v.stride_h = t.stride(1);
  }
  return v;
}

Affine make_affine(torch::Tensor const& t, int64_t heads, int64_t N) {
  Affine a{};
  a.ptr = t.numel() == 0 ? nullptr : t.data_ptr();
  a.dtype = dtype_from_tensor(t, true);
  a.stride_h = a.stride_n = 0;
  a.per_head = 0;
  if (a.dtype == DType::kNone) {
    return a;
  }
  TORCH_CHECK(t.dim() == 1 || t.dim() == 2, "affine tensors must be 1D or 2D");
  if (t.dim() == 1) {
    TORCH_CHECK(t.size(0) == N, "affine last dimension mismatch");
    TORCH_CHECK(t.stride(0) == 1, "affine tensor must be contiguous in the last dimension");
    a.stride_n = t.stride(0);
  } else {
    TORCH_CHECK(t.size(0) == heads && t.size(1) == N, "per-head affine shape mismatch");
    TORCH_CHECK(t.stride(1) == 1, "affine tensor must be contiguous in the last dimension");
    a.stride_h = t.stride(0);
    a.stride_n = t.stride(1);
    a.per_head = 1;
  }
  return a;
}

int threads_per_row_fwd(int64_t N) {
  if (N <= 64) return 8;
  if (N <= 128) return 16;
  if (N <= 3072) return 32;
  if (N <= 6144) return 64;
  if (N <= 16384) return 128;
  return 256;
}

int threads_per_row_bwd(int64_t N) {
  if (N <= 64) return 8;
  if (N <= 128) return 16;
  if (N <= 256) return 32;
  if (N <= 512) return 64;
  if (N <= 2048) return 128;
  if (N <= 4096) return 128;
  return 256;
}

int sm_count_for_device(int device) {
  thread_local int cached_device = -1;
  thread_local int cached_sm_count = 0;
  if (cached_device != device) {
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cached_device = device;
    cached_sm_count = prop.multiProcessorCount;
  }
  return cached_sm_count;
}

size_t smem_bytes(int num_threads, int threads_per_row) {
  int const rows_per_block = num_threads / threads_per_row;
  int const warps_per_row = (threads_per_row + 31) / 32;
  size_t bytes = sizeof(float) * size_t(rows_per_block) * size_t(warps_per_row);
  return bytes + 256;
}

void launch_kernel(void const* kernel,
                   dim3 grid,
                   dim3 block,
                   void** args,
                   size_t smem,
                   cudaStream_t stream) {
  TORCH_CHECK(smem <= size_t(std::numeric_limits<int>::max()), "dynamic smem too large");
  static std::mutex attr_mutex;
  static std::unordered_map<void const*, size_t> max_smem_cache;
  bool set_smem = false;
  {
    std::lock_guard<std::mutex> lock(attr_mutex);
    size_t& previous = max_smem_cache[kernel];
    if (smem > previous) {
      previous = smem;
      set_smem = true;
    }
  }
  if (set_smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(smem)));
  }
  TORCH_CHECK(cudaLaunchKernel(kernel, grid, block, args, smem, stream) == cudaSuccess,
              "cudaLaunchKernel failed");
}

void launch_kernel_cluster(void const* kernel,
                           dim3 grid,
                           dim3 block,
                           void** args,
                           size_t smem,
                           dim3 cluster_dim,
                           cudaStream_t stream) {
  TORCH_CHECK(smem <= size_t(std::numeric_limits<int>::max()), "dynamic smem too large");
  static std::mutex attr_mutex;
  static std::unordered_map<void const*, size_t> max_smem_cache;
  bool set_smem = false;
  {
    std::lock_guard<std::mutex> lock(attr_mutex);
    size_t& previous = max_smem_cache[kernel];
    if (smem > previous) {
      previous = smem;
      set_smem = true;
    }
  }
  if (set_smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(smem)));
  }
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_dim.x;
  attr.val.clusterDim.y = cluster_dim.y;
  attr.val.clusterDim.z = cluster_dim.z;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(kernel), args) ==
                  cudaSuccess,
              "cudaLaunchKernelExC cluster launch failed");
}

void launch_reduce_dw_partial(float* dw_partial,
                              float* dw,
                              int64_t partial_blocks,
                              int64_t N,
                              cudaStream_t stream,
                              bool use_column_reduce) {
  int64_t partial_blocks_arg = partial_blocks;
  int64_t N_arg = N;
  void* reduce_args[] = {&dw_partial, &dw, &partial_blocks_arg, &N_arg};
  dim3 reduce_block(256, 1, 1);
  if (use_column_reduce) {
    dim3 reduce_grid(static_cast<unsigned>(N), 1, 1);
    size_t reduce_smem = 8 * sizeof(float);
    auto reduce_kernel = rmsnorm_reduce_dw_partial_kernel;
    launch_kernel(reinterpret_cast<void const*>(reduce_kernel), reduce_grid, reduce_block,
                  reduce_args, reduce_smem, stream);
    return;
  }

  if (N == 2048 || N == 4096 || N == 8192 || N == 16384) {
    dim3 reduce_grid(static_cast<unsigned>((N + 15) / 16), 1, 1);
    size_t reduce_smem = 16 * 16 * sizeof(float);
    auto reduce_kernel = rmsnorm_reduce_dw_partial_tile16_kernel;
    launch_kernel(reinterpret_cast<void const*>(reduce_kernel), reduce_grid, reduce_block,
                  reduce_args, reduce_smem, stream);
    return;
  }

  dim3 reduce_grid(static_cast<unsigned>((N + 31) / 32), 1, 1);
  size_t reduce_smem = 8 * 32 * sizeof(float);
  auto reduce_kernel = rmsnorm_reduce_dw_partial_tile32_kernel;
  launch_kernel(reinterpret_cast<void const*>(reduce_kernel), reduce_grid, reduce_block,
                reduce_args, reduce_smem, stream);
}

template <int ThreadsPerRow, int NumThreads>
void launch_fwd_specialized(Tensor3 x,
                            Affine weight,
                            Affine bias,
                            Tensor3 residual,
                            Tensor3 out,
                            Tensor3 residual_out,
                            Tensor2 rstd,
                            int64_t rows_m,
                            int64_t heads,
                            int64_t N,
                            float eps,
                            cudaStream_t stream) {
  constexpr int kRowsPerBlock = NumThreads / ThreadsPerRow;
  dim3 grid(static_cast<unsigned>((rows_m * heads + kRowsPerBlock - 1) / kRowsPerBlock), 1, 1);
  dim3 block(NumThreads, 1, 1);
  size_t smem = smem_bytes(NumThreads, ThreadsPerRow);
  auto kernel = rmsnorm_fwd_kernel<ThreadsPerRow, NumThreads>;
  void* args[] = {
      &x, &weight, &bias, &residual, &out, &residual_out, &rstd, &rows_m, &heads, &N, &eps,
  };
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, args, smem, stream);
}

template <int NumThreads>
void dispatch_fwd_threads(int threads_per_row,
                          Tensor3 x,
                          Affine weight,
                          Affine bias,
                          Tensor3 residual,
                          Tensor3 out,
                          Tensor3 residual_out,
                          Tensor2 rstd,
                          int64_t rows_m,
                          int64_t heads,
                          int64_t N,
                          float eps,
                          cudaStream_t stream) {
  switch (threads_per_row) {
    case 8:
      launch_fwd_specialized<8, NumThreads>(x, weight, bias, residual, out, residual_out, rstd,
                                            rows_m, heads, N, eps, stream);
      break;
    case 16:
      launch_fwd_specialized<16, NumThreads>(x, weight, bias, residual, out, residual_out, rstd,
                                             rows_m, heads, N, eps, stream);
      break;
    case 32:
      launch_fwd_specialized<32, NumThreads>(x, weight, bias, residual, out, residual_out, rstd,
                                             rows_m, heads, N, eps, stream);
      break;
    case 64:
      launch_fwd_specialized<64, NumThreads>(x, weight, bias, residual, out, residual_out, rstd,
                                             rows_m, heads, N, eps, stream);
      break;
    case 128:
      launch_fwd_specialized<128, NumThreads>(x, weight, bias, residual, out, residual_out, rstd,
                                              rows_m, heads, N, eps, stream);
      break;
    case 256:
      if constexpr (NumThreads >= 256) {
        launch_fwd_specialized<256, NumThreads>(x, weight, bias, residual, out, residual_out, rstd,
                                                rows_m, heads, N, eps, stream);
      } else {
        TORCH_CHECK(false, "threads_per_row exceeds block size");
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported threads_per_row=", threads_per_row);
  }
}

template <typename T,
          int ThreadsPerRow,
          int NumThreads,
          int MaxVecs,
          bool HasResidual,
          bool HasRstd>
void launch_fwd_contig_specialized(T const* x,
                                   float const* weight,
                                   float const* bias,
                                   T const* residual,
                                   T* out,
                                   T* residual_out,
                                   float* rstd,
                                   int64_t M,
                                   int64_t N,
                                   float eps,
                                   cudaStream_t stream) {
  constexpr int kRowsPerBlock = NumThreads / ThreadsPerRow;
  dim3 grid(static_cast<unsigned>((M + kRowsPerBlock - 1) / kRowsPerBlock), 1, 1);
  dim3 block(NumThreads, 1, 1);
  size_t smem = smem_bytes(NumThreads, ThreadsPerRow);
  auto kernel = rmsnorm_fwd_contig_kernel<T, ThreadsPerRow, NumThreads, MaxVecs, HasResidual,
                                          HasRstd>;
  void* args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&bias),
      const_cast<T**>(&residual),
      &out,
      &residual_out,
      &rstd,
      &M,
      &N,
      &eps,
  };
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, args, smem, stream);
}

template <typename T, int ThreadsPerRow, int NumThreads, bool HasResidual, bool HasRstd>
bool dispatch_fwd_contig_maxvecs(int max_vecs,
                                 T const* x,
                                 float const* weight,
                                 float const* bias,
                                 T const* residual,
                                 T* out,
                                 T* residual_out,
                                 float* rstd,
                                 int64_t M,
                                 int64_t N,
                                 float eps,
                                 cudaStream_t stream) {
  switch (max_vecs) {
    case 1:
      launch_fwd_contig_specialized<T, ThreadsPerRow, NumThreads, 1, HasResidual, HasRstd>(
          x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
      return true;
    case 2:
      launch_fwd_contig_specialized<T, ThreadsPerRow, NumThreads, 2, HasResidual, HasRstd>(
          x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
      return true;
    case 4:
      launch_fwd_contig_specialized<T, ThreadsPerRow, NumThreads, 4, HasResidual, HasRstd>(
          x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
      return true;
    case 8:
      launch_fwd_contig_specialized<T, ThreadsPerRow, NumThreads, 8, HasResidual, HasRstd>(
          x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
      return true;
    case 16:
      launch_fwd_contig_specialized<T, ThreadsPerRow, NumThreads, 16, HasResidual, HasRstd>(
          x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
      return true;
    case 32:
      launch_fwd_contig_specialized<T, ThreadsPerRow, NumThreads, 32, HasResidual, HasRstd>(
          x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
      return true;
    default:
      return false;
  }
}

template <typename T, int ThreadsPerRow, int NumThreads>
bool dispatch_fwd_contig_flags(bool has_residual,
                               bool has_rstd,
                               int max_vecs,
                               T const* x,
                               float const* weight,
                               float const* bias,
                               T const* residual,
                               T* out,
                               T* residual_out,
                               float* rstd,
                               int64_t M,
                               int64_t N,
                               float eps,
                               cudaStream_t stream) {
  if (has_residual) {
    if (has_rstd) {
      return dispatch_fwd_contig_maxvecs<T, ThreadsPerRow, NumThreads, true, true>(
          max_vecs, x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
    }
    return dispatch_fwd_contig_maxvecs<T, ThreadsPerRow, NumThreads, true, false>(
        max_vecs, x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
  }
  if (has_rstd) {
    return dispatch_fwd_contig_maxvecs<T, ThreadsPerRow, NumThreads, false, true>(
        max_vecs, x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
  }
  return dispatch_fwd_contig_maxvecs<T, ThreadsPerRow, NumThreads, false, false>(
      max_vecs, x, weight, bias, residual, out, residual_out, rstd, M, N, eps, stream);
}

template <typename T, int NumThreads>
bool dispatch_fwd_contig_threads(int threads_per_row,
                                 int max_vecs,
                                 bool has_residual,
                                 bool has_rstd,
                                 T const* x,
                                 float const* weight,
                                 float const* bias,
                                 T const* residual,
                                 T* out,
                                 T* residual_out,
                                 float* rstd,
                                 int64_t M,
                                 int64_t N,
                                 float eps,
                                 cudaStream_t stream) {
  switch (threads_per_row) {
    case 8:
      return dispatch_fwd_contig_flags<T, 8, NumThreads>(has_residual, has_rstd, max_vecs, x,
                                                         weight, bias, residual, out,
                                                         residual_out, rstd, M, N, eps, stream);
    case 16:
      return dispatch_fwd_contig_flags<T, 16, NumThreads>(has_residual, has_rstd, max_vecs, x,
                                                          weight, bias, residual, out,
                                                          residual_out, rstd, M, N, eps, stream);
    case 32:
      return dispatch_fwd_contig_flags<T, 32, NumThreads>(has_residual, has_rstd, max_vecs, x,
                                                          weight, bias, residual, out,
                                                          residual_out, rstd, M, N, eps, stream);
    case 64:
      return dispatch_fwd_contig_flags<T, 64, NumThreads>(has_residual, has_rstd, max_vecs, x,
                                                          weight, bias, residual, out,
                                                          residual_out, rstd, M, N, eps, stream);
    case 128:
      return dispatch_fwd_contig_flags<T, 128, NumThreads>(has_residual, has_rstd, max_vecs, x,
                                                           weight, bias, residual, out,
                                                           residual_out, rstd, M, N, eps, stream);
    case 256:
      if constexpr (NumThreads >= 256) {
        return dispatch_fwd_contig_flags<T, 256, NumThreads>(has_residual, has_rstd, max_vecs, x,
                                                             weight, bias, residual, out,
                                                             residual_out, rstd, M, N, eps, stream);
      }
      return false;
    default:
      return false;
  }
}

template <typename T>
bool try_launch_fwd_contig(torch::Tensor const& x_t,
                           torch::Tensor const& weight_t,
                           torch::Tensor const& bias_t,
                           torch::Tensor const& residual_t,
                           torch::Tensor const& out_t,
                           torch::Tensor const& residual_out_t,
                           torch::Tensor const& rstd_t,
                           int threads_per_row,
                           int num_threads,
                           double eps,
                           cudaStream_t stream) {
  bool const has_residual = residual_t.numel() != 0;
  bool const has_rstd = rstd_t.numel() != 0;
  if (bias_t.numel() != 0) {
    return false;
  }
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (has_residual) {
    if (residual_t.scalar_type() != x_t.scalar_type() ||
        residual_out_t.scalar_type() != x_t.scalar_type() ||
        !residual_t.is_contiguous() || !residual_out_t.is_contiguous() ||
        residual_t.sizes() != x_t.sizes() || residual_out_t.sizes() != x_t.sizes()) {
      return false;
    }
  } else if (residual_out_t.numel() != 0) {
    return false;
  }
  if (has_rstd &&
      (rstd_t.scalar_type() != at::ScalarType::Float || !rstd_t.is_contiguous() ||
       rstd_t.dim() != 1 || rstd_t.size(0) != M)) {
    return false;
  }
  constexpr int kVec = Vec128<T>::kElements;
  if (N % (int64_t(kVec) * threads_per_row) != 0) {
    return false;
  }
  int const max_vecs = int(N / (int64_t(kVec) * threads_per_row));
  if (max_vecs < 1 || max_vecs > 32) {
    return false;
  }
  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  float const* bias = nullptr;
  T const* residual =
      has_residual ? reinterpret_cast<T const*>(residual_t.const_data_ptr()) : nullptr;
  auto* out = reinterpret_cast<T*>(out_t.data_ptr());
  T* residual_out = has_residual ? reinterpret_cast<T*>(residual_out_t.data_ptr()) : nullptr;
  float* rstd = has_rstd ? rstd_t.data_ptr<float>() : nullptr;
  if (num_threads == 128) {
    return dispatch_fwd_contig_threads<T, 128>(threads_per_row, max_vecs, has_residual,
                                               has_rstd, x, weight, bias, residual, out,
                                               residual_out, rstd, M, N,
                                               static_cast<float>(eps), stream);
  }
  if (num_threads == 256) {
    return dispatch_fwd_contig_threads<T, 256>(threads_per_row, max_vecs, has_residual,
                                               has_rstd, x, weight, bias, residual, out,
                                               residual_out, rstd, M, N,
                                               static_cast<float>(eps), stream);
  }
  return false;
}

bool try_launch_fwd_contig_fp32_preload_w(torch::Tensor const& x_t,
                                          torch::Tensor const& weight_t,
                                          torch::Tensor const& out_t,
                                          double eps,
                                          cudaStream_t stream) {
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 4096 && N != 8192) {
    return false;
  }

  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto* out = out_t.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  float eps_arg = static_cast<float>(eps);
  void* args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      &out,
      &M_arg,
      &N_arg,
      &eps_arg,
  };
  if (N == 4096) {
    constexpr int kThreadsPerRow512 = 512;
    constexpr int kNumThreads512 = 512;
    dim3 grid(static_cast<unsigned>(M), 1, 1);
    dim3 block(kNumThreads512, 1, 1);
    size_t smem = smem_bytes(kNumThreads512, kThreadsPerRow512);
    auto kernel =
        rmsnorm_fwd_contig_fp32_preload_w_kernel<kThreadsPerRow512, kNumThreads512, 2>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, args, smem, stream);
    return true;
  }
  constexpr int kThreadsPerRow = 128;
  constexpr int kMaxVecs = 16;
  dim3 grid(static_cast<unsigned>(M), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = size_t(kThreadsPerRow) * size_t(kMaxVecs) * sizeof(uint4) +
                size_t(kThreadsPerRow / 32) * sizeof(float) + sizeof(uint64_t);
  auto kernel = rmsnorm_fwd_contig_fp32_smem_async_preload_w_kernel<kThreadsPerRow, kMaxVecs>;
  void* bulk_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      &out,
      &eps_arg,
  };
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, bulk_args, smem, stream);
  return true;
}

template <typename T, int kMaxVecs>
bool try_launch_fwd_contig_stream_impl(torch::Tensor const& x_t,
                                       torch::Tensor const& weight_t,
                                       torch::Tensor const& out_t,
                                       double eps,
                                       cudaStream_t stream) {
  constexpr int kThreadsPerRow = 256;
  constexpr int64_t kSupportedN =
      int64_t(kThreadsPerRow) * Vec128<T>::kElements * kMaxVecs;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != kSupportedN) {
    return false;
  }

  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto* out = reinterpret_cast<T*>(out_t.data_ptr());
  int64_t M_arg = M;
  int64_t N_arg = N;
  float eps_arg = static_cast<float>(eps);
  dim3 grid(static_cast<unsigned>(M), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = smem_bytes(kThreadsPerRow, kThreadsPerRow);
  auto kernel = rmsnorm_fwd_contig_stream_kernel<T, kThreadsPerRow, kMaxVecs>;
  void* args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      &out,
      &M_arg,
      &N_arg,
      &eps_arg,
  };
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, args, smem, stream);
  return true;
}

template <typename T>
bool try_launch_fwd_contig_stream(torch::Tensor const& x_t,
                                  torch::Tensor const& weight_t,
                                  torch::Tensor const& out_t,
                                  double eps,
                                  cudaStream_t stream) {
  if (x_t.size(1) == int64_t(256) * Vec128<T>::kElements * 64) {
    return try_launch_fwd_contig_stream_impl<T, 64>(x_t, weight_t, out_t, eps, stream);
  }
  return try_launch_fwd_contig_stream_impl<T, 32>(x_t, weight_t, out_t, eps, stream);
}

template <typename T,
          int kClusterN,
          int kMaxVecs,
          bool kCacheXSmem = false,
          int kRegCacheVecs = 0,
          int kTailSmemStartVec = 0,
          int kThreadsPerRow = 256>
bool try_launch_fwd_contig_cluster_impl(torch::Tensor const& x_t,
                                        torch::Tensor const& weight_t,
                                        torch::Tensor const& out_t,
                                        double eps,
                                        cudaStream_t stream) {
  constexpr int64_t kSupportedN =
      int64_t(kClusterN) * kThreadsPerRow * Vec128<T>::kElements * kMaxVecs;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != kSupportedN) {
    return false;
  }

  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto* out = reinterpret_cast<T*>(out_t.data_ptr());
  int64_t M_arg = M;
  int64_t N_arg = N;
  float eps_arg = static_cast<float>(eps);
  dim3 grid(static_cast<unsigned>(M * kClusterN), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = 32 * sizeof(float) + (kThreadsPerRow / 32) * kClusterN * sizeof(float) +
                sizeof(uint64_t) + 256;
  if constexpr (kCacheXSmem) {
    smem += sizeof(uint4) * size_t(kThreadsPerRow) * size_t(kMaxVecs);
  } else if constexpr (kTailSmemStartVec > 0) {
    smem += sizeof(uint4) * size_t(kThreadsPerRow) *
            size_t(kMaxVecs - kTailSmemStartVec);
  }
  void* args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      &out,
      &M_arg,
      &N_arg,
      &eps_arg,
  };
  auto kernel =
      rmsnorm_fwd_cluster_kernel<T, kClusterN, kThreadsPerRow, kMaxVecs, kCacheXSmem,
                                 kRegCacheVecs, kTailSmemStartVec>;
  launch_kernel_cluster(reinterpret_cast<void const*>(kernel), grid, block, args, smem,
                        dim3(kClusterN, 1, 1), stream);
  return true;
}

template <typename T>
bool try_launch_fwd_contig_cluster(torch::Tensor const& x_t,
                                   torch::Tensor const& weight_t,
                                   torch::Tensor const& out_t,
                                   double eps,
                                   cudaStream_t stream) {
  int64_t const N = x_t.size(1);
  if constexpr (sizeof(T) == 2) {
    if (N == 32768) {
      return try_launch_fwd_contig_cluster_impl<T, 2, 8, false, 6>(
          x_t, weight_t, out_t, eps, stream);
    }
    if (N == 65536) {
      // Wide half rows benefit from staging the output-pass tail in smem.
      // FP16 was fastest with one middle reload; BF16 preferred staging all
      // non-register vectors for this width.
      if constexpr (std::is_same_v<T, half>) {
        return try_launch_fwd_contig_cluster_impl<T, 4, 8, false, 4, 5>(
            x_t, weight_t, out_t, eps, stream);
      } else {
        return try_launch_fwd_contig_cluster_impl<T, 4, 8, false, 4, 4>(
            x_t, weight_t, out_t, eps, stream);
      }
    }
    if (N == 131072) {
      return try_launch_fwd_contig_cluster_impl<T, 4, 16, false, 8, 8>(
          x_t, weight_t, out_t, eps, stream);
    }
    if (N == 262144) {
      // Same tail-smem strategy as 128K; FP16 is slightly register-sensitive,
      // while BF16 wins by caching the full prefix through output.
      if constexpr (std::is_same_v<T, half>) {
        return try_launch_fwd_contig_cluster_impl<T, 8, 16, false, 6, 8>(
            x_t, weight_t, out_t, eps, stream);
      } else {
        return try_launch_fwd_contig_cluster_impl<T, 8, 16, false, 8, 8>(
            x_t, weight_t, out_t, eps, stream);
      }
    }
  } else if constexpr (sizeof(T) == 4) {
    if (N == 32768) {
      return try_launch_fwd_contig_cluster_impl<T, 4, 8, false, 8>(
          x_t, weight_t, out_t, eps, stream);
    }
    if (N == 65536) {
      return try_launch_fwd_contig_cluster_impl<T, 4, 16, false, 16>(
          x_t, weight_t, out_t, eps, stream);
    }
    if (N == 131072) {
      return try_launch_fwd_contig_cluster_impl<T, 8, 16, false, 16>(
          x_t, weight_t, out_t, eps, stream);
    }
    if (N == 262144) {
      // 512-thread B200 shape: vectors 0..9 stay in registers through output,
      // and 10..15 are staged in smem. This avoids the second x read without
      // the occupancy loss of caching the whole CTA tile.
      return try_launch_fwd_contig_cluster_impl<T, 8, 16, false, 10, 10, 512>(
          x_t, weight_t, out_t, eps, stream);
    }
  }
  return false;
}

template <typename T, int kMaxVecs>
bool try_launch_fwd_contig_split_impl(torch::Tensor const& x_t,
                                      torch::Tensor const& weight_t,
                                      torch::Tensor const& out_t,
                                      double eps,
                                      cudaStream_t stream) {
  constexpr int kThreadsPerRow = 256;
  constexpr int64_t kSegmentCols = int64_t(kThreadsPerRow) * Vec128<T>::kElements * kMaxVecs;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N < kSegmentCols * 2 || N % kSegmentCols != 0) {
    return false;
  }
  int64_t const slices = N / kSegmentCols;
  if (slices < 2 || slices > 32) {
    return false;
  }

  auto sumsq_partial = torch::empty({M, slices}, x_t.options().dtype(torch::kFloat32));
  auto rstd_tmp = torch::empty({M}, x_t.options().dtype(torch::kFloat32));

  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  T const* residual = nullptr;
  T* residual_out = nullptr;
  auto* out = reinterpret_cast<T*>(out_t.data_ptr());
  auto* sumsq = sumsq_partial.data_ptr<float>();
  auto* rstd = rstd_tmp.data_ptr<float>();

  dim3 grid(static_cast<unsigned>(M), static_cast<unsigned>(slices), 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = smem_bytes(kThreadsPerRow, kThreadsPerRow);
  int64_t M_arg = M;
  int64_t N_arg = N;
  int64_t slices_arg = slices;
  int64_t segment_arg = kSegmentCols;

  auto sumsq_kernel = rmsnorm_fwd_split_sumsq_kernel<T, kThreadsPerRow, kMaxVecs>;
  void* sumsq_args[] = {
      const_cast<T**>(&x),
      const_cast<T**>(&residual),
      &sumsq,
      &M_arg,
      &N_arg,
      &slices_arg,
      &segment_arg,
  };
  launch_kernel(reinterpret_cast<void const*>(sumsq_kernel), grid, block, sumsq_args, smem,
                stream);

  dim3 reduce_grid(static_cast<unsigned>((M + 255) / 256), 1, 1);
  dim3 reduce_block(256, 1, 1);
  float eps_arg = static_cast<float>(eps);
  void* reduce_args[] = {&sumsq, &rstd, &M_arg, &N_arg, &slices_arg, &eps_arg};
  auto reduce_kernel = rmsnorm_fwd_reduce_sumsq_kernel;
  launch_kernel(reinterpret_cast<void const*>(reduce_kernel), reduce_grid, reduce_block,
                reduce_args, 0, stream);

  auto output_kernel = rmsnorm_fwd_split_output_kernel<T, kThreadsPerRow, kMaxVecs>;
  void* output_args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      const_cast<T**>(&residual),
      &rstd,
      &out,
      &residual_out,
      &M_arg,
      &N_arg,
      &segment_arg,
  };
  launch_kernel(reinterpret_cast<void const*>(output_kernel), grid, block, output_args, 0,
                stream);
  return true;
}

template <typename T>
bool try_launch_fwd_contig_split(torch::Tensor const& x_t,
                                 torch::Tensor const& weight_t,
                                 torch::Tensor const& out_t,
                                 double eps,
                                 cudaStream_t stream) {
  return try_launch_fwd_contig_split_impl<T, 8>(x_t, weight_t, out_t, eps, stream);
}

template <typename T>
bool try_launch_fwd_contig_split_residual(torch::Tensor const& x_t,
                                          torch::Tensor const& weight_t,
                                          torch::Tensor const& residual_t,
                                          torch::Tensor const& out_t,
                                          torch::Tensor const& residual_out_t,
                                          torch::Tensor const& rstd_t,
                                          double eps,
                                          cudaStream_t stream) {
  constexpr int kThreadsPerRow = 256;
  constexpr int kMaxVecs = 8;
  constexpr int64_t kSegmentCols =
      int64_t(kThreadsPerRow) * Vec128<T>::kElements * kMaxVecs;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N < 65536 || N < kSegmentCols * 2 || N % kSegmentCols != 0) {
    return false;
  }
  int64_t const slices = N / kSegmentCols;
  if (slices < 2 || slices > 32) {
    return false;
  }

  auto sumsq_partial = torch::empty({M, slices}, x_t.options().dtype(torch::kFloat32));
  auto rstd_storage = rstd_t.numel() != 0
                          ? rstd_t
                          : torch::empty({M}, x_t.options().dtype(torch::kFloat32));

  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* residual = reinterpret_cast<T const*>(residual_t.const_data_ptr());
  auto* out = reinterpret_cast<T*>(out_t.data_ptr());
  auto* residual_out = reinterpret_cast<T*>(residual_out_t.data_ptr());
  auto* sumsq = sumsq_partial.data_ptr<float>();
  auto* rstd = rstd_storage.data_ptr<float>();

  dim3 grid(static_cast<unsigned>(M), static_cast<unsigned>(slices), 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = smem_bytes(kThreadsPerRow, kThreadsPerRow);
  int64_t M_arg = M;
  int64_t N_arg = N;
  int64_t slices_arg = slices;
  int64_t segment_arg = kSegmentCols;

  auto sumsq_kernel =
      rmsnorm_fwd_split_sumsq_kernel<T, kThreadsPerRow, kMaxVecs, true>;
  void* sumsq_args[] = {
      const_cast<T**>(&x),
      const_cast<T**>(&residual),
      &sumsq,
      &M_arg,
      &N_arg,
      &slices_arg,
      &segment_arg,
  };
  launch_kernel(reinterpret_cast<void const*>(sumsq_kernel), grid, block, sumsq_args, smem,
                stream);

  dim3 reduce_grid(static_cast<unsigned>((M + 255) / 256), 1, 1);
  dim3 reduce_block(256, 1, 1);
  float eps_arg = static_cast<float>(eps);
  void* reduce_args[] = {&sumsq, &rstd, &M_arg, &N_arg, &slices_arg, &eps_arg};
  auto reduce_kernel = rmsnorm_fwd_reduce_sumsq_kernel;
  launch_kernel(reinterpret_cast<void const*>(reduce_kernel), reduce_grid, reduce_block,
                reduce_args, 0, stream);

  auto output_kernel =
      rmsnorm_fwd_split_output_kernel<T, kThreadsPerRow, kMaxVecs, true>;
  void* output_args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      const_cast<T**>(&residual),
      &rstd,
      &out,
      &residual_out,
      &M_arg,
      &N_arg,
      &segment_arg,
  };
  launch_kernel(reinterpret_cast<void const*>(output_kernel), grid, block, output_args, 0,
                stream);
  return true;
}

int fast_bwd_partial_blocks(int64_t N, int device) {
  int const sm_count = sm_count_for_device(device);
  int multiple = 1;
  if (N <= 256) {
    multiple = 16;
  } else if (N <= 1024) {
    multiple = 8;
  } else if (N <= 2048) {
    multiple = 4;
  } else if (N <= 4096) {
    multiple = 3;
  }
  int blocks = sm_count * multiple;
  if (N > 8192 && N <= 16384) {
    blocks = sm_count;
  } else if (N > 131072) {
    blocks = std::max(sm_count / 4, 1);
  } else if (N > 32768) {
    blocks = std::max(sm_count / 2, 1);
  } else if (N > 16384) {
    blocks = sm_count;
  }
  return blocks;
}

template <typename T,
          int ThreadsPerRow,
          int MaxVecs,
          int StaticN = 0,
          bool HasDresidualOut = false>
void launch_bwd_contig_partial_specialized(T const* x,
                                           float const* weight,
                                           T const* dout,
                                           float const* rstd,
                                           T* dx,
                                           T const* dresidual_out,
                                           float* dw_partial,
                                           int partial_blocks,
                                           int64_t M,
                                           int64_t N,
                                           cudaStream_t stream) {
  dim3 grid(static_cast<unsigned>(partial_blocks), 1, 1);
  dim3 block(ThreadsPerRow, 1, 1);
  size_t smem = smem_bytes(ThreadsPerRow, ThreadsPerRow);
  auto kernel =
      rmsnorm_bwd_contig_partial_kernel<T, ThreadsPerRow, MaxVecs, StaticN, HasDresidualOut>;
  int64_t partial_blocks_arg = partial_blocks;
  void* args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      const_cast<T**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<T**>(&dresidual_out),
      &dw_partial,
      &M,
      &N,
  };
  (void)partial_blocks_arg;
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, args, smem, stream);
}

template <typename T, int ThreadsPerRow>
bool dispatch_bwd_contig_maxvecs(int max_vecs,
                                 T const* x,
                                 float const* weight,
                                 T const* dout,
                                 float const* rstd,
                                 T* dx,
                                 float* dw_partial,
                                 int partial_blocks,
                                 int64_t M,
                                 int64_t N,
                                 cudaStream_t stream) {
  switch (max_vecs) {
    case 1:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 1>(
          x, weight, dout, rstd, dx, nullptr, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 2:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 2>(
          x, weight, dout, rstd, dx, nullptr, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 4:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 4>(
          x, weight, dout, rstd, dx, nullptr, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 8:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 8>(
          x, weight, dout, rstd, dx, nullptr, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 16:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 16>(
          x, weight, dout, rstd, dx, nullptr, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 32:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 32>(
          x, weight, dout, rstd, dx, nullptr, dw_partial, partial_blocks, M, N, stream);
      return true;
    default:
      return false;
  }
}

template <typename T, int ThreadsPerRow>
bool dispatch_bwd_contig_residual_maxvecs(int max_vecs,
                                          T const* x,
                                          float const* weight,
                                          T const* dout,
                                          float const* rstd,
                                          T* dx,
                                          T const* dresidual_out,
                                          float* dw_partial,
                                          int partial_blocks,
                                          int64_t M,
                                          int64_t N,
                                          cudaStream_t stream) {
  switch (max_vecs) {
    case 1:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 1, 0, true>(
          x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 2:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 2, 0, true>(
          x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 4:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 4, 0, true>(
          x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 8:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 8, 0, true>(
          x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 16:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 16, 0, true>(
          x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N, stream);
      return true;
    case 32:
      launch_bwd_contig_partial_specialized<T, ThreadsPerRow, 32, 0, true>(
          x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N, stream);
      return true;
    default:
      return false;
  }
}

template <typename T>
bool try_launch_bwd_contig(torch::Tensor const& x_t,
                           torch::Tensor const& weight_t,
                           torch::Tensor const& dout_t,
                           torch::Tensor const& rstd_t,
                           torch::Tensor const& dx_t,
                           torch::Tensor const& dw_t,
                           int threads_per_row,
                           cudaStream_t stream) {
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  constexpr int kVec = Vec128<T>::kElements;
  if (N % (int64_t(kVec) * threads_per_row) != 0) {
    return false;
  }
  int const max_vecs = int(N / (int64_t(kVec) * threads_per_row));
  if (max_vecs < 1 || max_vecs > 32) {
    return false;
  }

  int partial_blocks = fast_bwd_partial_blocks(N, x_t.get_device());
  if constexpr (sizeof(T) == 4) {
    if (N == 4096) {
      partial_blocks = std::max(sm_count_for_device(x_t.get_device()) * 2, 1);
    }
  }
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = reinterpret_cast<T const*>(dout_t.const_data_ptr());
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = reinterpret_cast<T*>(dx_t.data_ptr());
  T const* dresidual_out = nullptr;
  auto* dw_partial = partial.data_ptr<float>();
  bool launched = false;
  switch (threads_per_row) {
    case 8:
      if (!launched) {
        launched = dispatch_bwd_contig_maxvecs<T, 8>(max_vecs, x, weight, dout, rstd, dx,
                                                     dw_partial, partial_blocks, M, N, stream);
      }
      break;
    case 16:
      if (!launched) {
        launched = dispatch_bwd_contig_maxvecs<T, 16>(max_vecs, x, weight, dout, rstd, dx,
                                                      dw_partial, partial_blocks, M, N, stream);
      }
      break;
    case 32:
      if (!launched) {
        launched = dispatch_bwd_contig_maxvecs<T, 32>(max_vecs, x, weight, dout, rstd, dx,
                                                      dw_partial, partial_blocks, M, N, stream);
      }
      break;
    case 64:
      if (!launched) {
        launched = dispatch_bwd_contig_maxvecs<T, 64>(max_vecs, x, weight, dout, rstd, dx,
                                                      dw_partial, partial_blocks, M, N, stream);
      }
      break;
    case 128:
      if (!launched) {
        launched = dispatch_bwd_contig_maxvecs<T, 128>(max_vecs, x, weight, dout, rstd, dx,
                                                       dw_partial, partial_blocks, M, N, stream);
      }
      break;
    case 256:
      if (!launched) {
        launched = dispatch_bwd_contig_maxvecs<T, 256>(max_vecs, x, weight, dout, rstd, dx,
                                                       dw_partial, partial_blocks, M, N, stream);
      }
      break;
    default:
      launched = false;
      break;
  }
  if (!launched) {
    return false;
  }

  float* dw = dw_t.data_ptr<float>();
  bool const use_column_reduce = N <= 1024;
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, use_column_reduce);
  return true;
}

template <typename T>
bool try_launch_bwd_contig_residual(torch::Tensor const& x_t,
                                    torch::Tensor const& weight_t,
                                    torch::Tensor const& dout_t,
                                    torch::Tensor const& rstd_t,
                                    torch::Tensor const& dresidual_out_t,
                                    torch::Tensor const& dx_t,
                                    torch::Tensor const& dw_t,
                                    int threads_per_row,
                                    cudaStream_t stream) {
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  constexpr int kVec = Vec128<T>::kElements;
  if (N % (int64_t(kVec) * threads_per_row) != 0) {
    return false;
  }
  int const max_vecs = int(N / (int64_t(kVec) * threads_per_row));
  if (max_vecs < 1 || max_vecs > 32) {
    return false;
  }

  int partial_blocks = fast_bwd_partial_blocks(N, x_t.get_device());
  if constexpr (sizeof(T) == 4) {
    if (N == 4096) {
      partial_blocks = std::max(sm_count_for_device(x_t.get_device()) * 2, 1);
    }
  }
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = reinterpret_cast<T const*>(dout_t.const_data_ptr());
  auto const* rstd = rstd_t.data_ptr<float>();
  auto const* dresidual_out = reinterpret_cast<T const*>(dresidual_out_t.const_data_ptr());
  auto* dx = reinterpret_cast<T*>(dx_t.data_ptr());
  auto* dw_partial = partial.data_ptr<float>();
  bool launched = false;
  switch (threads_per_row) {
    case 8:
      launched = dispatch_bwd_contig_residual_maxvecs<T, 8>(
          max_vecs, x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N,
          stream);
      break;
    case 16:
      launched = dispatch_bwd_contig_residual_maxvecs<T, 16>(
          max_vecs, x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N,
          stream);
      break;
    case 32:
      launched = dispatch_bwd_contig_residual_maxvecs<T, 32>(
          max_vecs, x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N,
          stream);
      break;
    case 64:
      launched = dispatch_bwd_contig_residual_maxvecs<T, 64>(
          max_vecs, x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N,
          stream);
      break;
    case 128:
      launched = dispatch_bwd_contig_residual_maxvecs<T, 128>(
          max_vecs, x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N,
          stream);
      break;
    case 256:
      launched = dispatch_bwd_contig_residual_maxvecs<T, 256>(
          max_vecs, x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N,
          stream);
      break;
    default:
      launched = false;
      break;
  }
  if (!launched) {
    return false;
  }

  float* dw = dw_t.data_ptr<float>();
  bool const use_column_reduce = N <= 1024;
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, use_column_reduce);
  return true;
}

bool try_launch_bwd_contig_fp32_32768_wide(torch::Tensor const& x_t,
                                           torch::Tensor const& weight_t,
                                           torch::Tensor const& dout_t,
                                           torch::Tensor const& rstd_t,
                                           torch::Tensor const& dx_t,
                                           torch::Tensor const& dw_t,
                                           cudaStream_t stream) {
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 32768) {
    return false;
  }
  int const partial_blocks = std::max(sm_count_for_device(x_t.get_device()) * 2, 1);
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = dout_t.const_data_ptr<float>();
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = dx_t.data_ptr<float>();
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks), 1, 1);
  dim3 block(512, 1, 1);
  size_t smem = smem_bytes(512, 512);
  auto kernel = rmsnorm_bwd_contig_fp32_stream_kernel<512, 16>;
  void* kernel_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

bool try_launch_bwd_contig_fp32_16384_smem_dw(torch::Tensor const& x_t,
                                              torch::Tensor const& weight_t,
                                              torch::Tensor const& dout_t,
                                              torch::Tensor const& rstd_t,
                                              torch::Tensor const& dx_t,
                                              torch::Tensor const& dw_t,
                                              cudaStream_t stream) {
  constexpr int kThreadsPerRow = 256;
  constexpr int kMaxVecs = 16;
  constexpr int kVec = Vec128<float>::kElements;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 16384) {
    return false;
  }

  int const partial_blocks = std::max(sm_count_for_device(x_t.get_device()) * 2, 1);
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));

  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = dout_t.const_data_ptr<float>();
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = dx_t.data_ptr<float>();
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = 32 * sizeof(float) +
                size_t(kThreadsPerRow) * size_t(kMaxVecs * kVec) * sizeof(float) + 256;
  void* kernel_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  auto kernel = rmsnorm_bwd_contig_fp32_smem_dw_kernel<kThreadsPerRow, kMaxVecs>;
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

bool try_launch_bwd_contig_fp32_16384_cluster_reg_residual(
    torch::Tensor const& x_t,
    torch::Tensor const& weight_t,
    torch::Tensor const& dout_t,
    torch::Tensor const& rstd_t,
    torch::Tensor const& dresidual_out_t,
    torch::Tensor const& dx_t,
    torch::Tensor const& dw_t,
    cudaStream_t stream) {
  constexpr int kClusterN = 8;
  constexpr int kThreadsPerRow = 128;
  constexpr int kMaxVecs = 4;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 16384) {
    return false;
  }

  // This residual fp32 16K path is memory-throughput limited; a deeper persistent
  // queue improves scheduling enough to offset the larger final dW reduction.
  int const partial_blocks = std::max(sm_count_for_device(x_t.get_device()) * 6, 1);
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));

  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = dout_t.const_data_ptr<float>();
  auto const* rstd = rstd_t.data_ptr<float>();
  auto const* dresidual_out = dresidual_out_t.const_data_ptr<float>();
  auto* dx = dx_t.data_ptr<float>();
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks * kClusterN), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = 32 * sizeof(float) + (kThreadsPerRow / 32) * kClusterN * sizeof(float) +
                sizeof(uint64_t) + 256;
  void* kernel_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<float**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  auto kernel =
      rmsnorm_bwd_contig_fp32_cluster_reg_kernel<kClusterN, kThreadsPerRow, kMaxVecs, false,
                                                 false, true, true, false, 2, true, false, true,
                                                 true>;
  launch_kernel_cluster(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem,
                        dim3(kClusterN, 1, 1), stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

bool try_launch_bwd_contig_fp32_cp_async_residual(
    torch::Tensor const& x_t,
    torch::Tensor const& weight_t,
    torch::Tensor const& dout_t,
    torch::Tensor const& rstd_t,
    torch::Tensor const& dresidual_out_t,
    torch::Tensor const& dx_t,
    torch::Tensor const& dw_t,
    cudaStream_t stream) {
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  int threads_per_row = 0;
  int max_vecs = 0;
  if (N == 512) {
    threads_per_row = 64;
    max_vecs = 2;
  } else if (N == 1024) {
    threads_per_row = 128;
    max_vecs = 2;
  } else if (N == 2048) {
    threads_per_row = 128;
    max_vecs = 4;
  } else if (N == 4096) {
    threads_per_row = 256;
    max_vecs = 4;
  } else {
    return false;
  }

  int const partial_blocks =
      N == 4096 ? std::max(sm_count_for_device(x_t.get_device()) * 2, 1)
                : fast_bwd_partial_blocks(N, x_t.get_device());
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = dout_t.const_data_ptr<float>();
  auto const* rstd = rstd_t.data_ptr<float>();
  auto const* dresidual_out = dresidual_out_t.const_data_ptr<float>();
  auto* dx = dx_t.data_ptr<float>();
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks), 1, 1);
  dim3 block(static_cast<unsigned>(threads_per_row), 1, 1);
  size_t smem = 6 * size_t(threads_per_row) * size_t(max_vecs) * sizeof(uint4) +
                32 * sizeof(float) + 256;
  void* kernel_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<float**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  if (N == 512) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<float, 64, 2, 512, true>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  } else if (N == 1024) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<float, 128, 2, 1024, true>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  } else if (N == 2048) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<float, 128, 4, 2048, true>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  } else if (N == 4096) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<float, 256, 4, 4096, true>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  }

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, N <= 1024);
  return true;
}

bool try_launch_bwd_contig_fp32_8192_cp_async(torch::Tensor const& x_t,
                                              torch::Tensor const& weight_t,
                                              torch::Tensor const& dout_t,
                                              torch::Tensor const& rstd_t,
                                              torch::Tensor const& dx_t,
                                              torch::Tensor const& dw_t,
                                              cudaStream_t stream) {
  constexpr int kThreadsPerRow = 256;
  constexpr int kMaxVecs = 8;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 8192) {
    return false;
  }

  int const partial_blocks = std::max(sm_count_for_device(x_t.get_device()) * 3, 1);
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = dout_t.const_data_ptr<float>();
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = dx_t.data_ptr<float>();
  float const* dresidual_out = nullptr;
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = 4 * size_t(kThreadsPerRow) * size_t(kMaxVecs) * sizeof(uint4) +
                32 * sizeof(float) + 256;
  void* kernel_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<float**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  auto kernel = rmsnorm_bwd_contig_cp_async_kernel<float, kThreadsPerRow, kMaxVecs, 8192>;
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

bool try_launch_bwd_contig_fp32_8192_cp_async_residual(torch::Tensor const& x_t,
                                                       torch::Tensor const& weight_t,
                                                       torch::Tensor const& dout_t,
                                                       torch::Tensor const& rstd_t,
                                                       torch::Tensor const& dresidual_out_t,
                                                       torch::Tensor const& dx_t,
                                                       torch::Tensor const& dw_t,
                                                       cudaStream_t stream) {
  constexpr int kThreadsPerRow = 256;
  constexpr int kMaxVecs = 8;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 8192) {
    return false;
  }

  int const partial_blocks = fast_bwd_partial_blocks(N, x_t.get_device());
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = dout_t.const_data_ptr<float>();
  auto const* rstd = rstd_t.data_ptr<float>();
  auto const* dresidual_out = dresidual_out_t.const_data_ptr<float>();
  auto* dx = dx_t.data_ptr<float>();
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = 6 * size_t(kThreadsPerRow) * size_t(kMaxVecs) * sizeof(uint4) +
                32 * sizeof(float) + 256;
  void* kernel_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<float**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  auto kernel =
      rmsnorm_bwd_contig_cp_async_kernel<float, kThreadsPerRow, kMaxVecs, 8192, true>;
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

template <typename T>
bool try_launch_bwd_contig_half_cp_async(torch::Tensor const& x_t,
                                         torch::Tensor const& weight_t,
                                         torch::Tensor const& dout_t,
                                         torch::Tensor const& rstd_t,
                                         torch::Tensor const& dx_t,
                                         torch::Tensor const& dw_t,
                                         cudaStream_t stream) {
  static_assert(sizeof(T) == 2);
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  int threads_per_row = 0;
  int max_vecs = 0;
  if (N == 4096) {
    threads_per_row = 256;
    max_vecs = 2;
  } else if (N == 8192) {
    threads_per_row = 256;
    max_vecs = 4;
  } else if (N == 16384) {
    threads_per_row = 512;
    max_vecs = 4;
  } else {
    return false;
  }

  int const partial_blocks =
      N == 8192 ? std::max(sm_count_for_device(x_t.get_device()) * 2, 1)
                : fast_bwd_partial_blocks(N, x_t.get_device());
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));

  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = reinterpret_cast<T const*>(dout_t.const_data_ptr());
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = reinterpret_cast<T*>(dx_t.data_ptr());
  T const* dresidual_out = nullptr;
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks), 1, 1);
  dim3 block(static_cast<unsigned>(threads_per_row), 1, 1);
  size_t smem = 2 * 2 * size_t(threads_per_row) * size_t(max_vecs) * sizeof(uint4) +
                32 * sizeof(float) + 256;
  void* kernel_args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      const_cast<T**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<T**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  if (N == 4096) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<T, 256, 2, 4096>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  } else if (N == 8192) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<T, 256, 4, 8192>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  } else if (N == 16384) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<T, 512, 4, 16384>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  }

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

template <typename T>
bool try_launch_bwd_contig_half_cp_async_residual(torch::Tensor const& x_t,
                                                  torch::Tensor const& weight_t,
                                                  torch::Tensor const& dout_t,
                                                  torch::Tensor const& rstd_t,
                                                  torch::Tensor const& dresidual_out_t,
                                                  torch::Tensor const& dx_t,
                                                  torch::Tensor const& dw_t,
                                                  cudaStream_t stream) {
  static_assert(sizeof(T) == 2);
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  int threads_per_row = 0;
  int max_vecs = 0;
  if (N == 512) {
    threads_per_row = 64;
    max_vecs = 1;
  } else if (N == 4096) {
    threads_per_row = 256;
    max_vecs = 2;
  } else if (N == 16384) {
    threads_per_row = 512;
    max_vecs = 4;
  } else {
    return false;
  }

  int const partial_blocks = fast_bwd_partial_blocks(N, x_t.get_device());
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));

  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = reinterpret_cast<T const*>(dout_t.const_data_ptr());
  auto const* rstd = rstd_t.data_ptr<float>();
  auto const* dresidual_out = reinterpret_cast<T const*>(dresidual_out_t.const_data_ptr());
  auto* dx = reinterpret_cast<T*>(dx_t.data_ptr());
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks), 1, 1);
  dim3 block(static_cast<unsigned>(threads_per_row), 1, 1);
  size_t smem = 3 * 2 * size_t(threads_per_row) * size_t(max_vecs) * sizeof(uint4) +
                32 * sizeof(float) + 256;
  void* kernel_args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      const_cast<T**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<T**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  if (N == 512) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<T, 64, 1, 512, true>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  } else if (N == 4096) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<T, 256, 2, 4096, true>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  } else if (N == 16384) {
    auto kernel = rmsnorm_bwd_contig_cp_async_kernel<T, 512, 4, 16384, true>;
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem, stream);
  }

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

template <typename T>
bool try_launch_bwd_contig_half_8192_tpr1024(torch::Tensor const& x_t,
                                             torch::Tensor const& weight_t,
                                             torch::Tensor const& dout_t,
                                             torch::Tensor const& rstd_t,
                                             torch::Tensor const& dx_t,
                                             torch::Tensor const& dw_t,
                                             cudaStream_t stream) {
  static_assert(sizeof(T) == 2);
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 8192) {
    return false;
  }

  int const partial_blocks = fast_bwd_partial_blocks(N, x_t.get_device());
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = reinterpret_cast<T const*>(dout_t.const_data_ptr());
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = reinterpret_cast<T*>(dx_t.data_ptr());
  auto* dw_partial = partial.data_ptr<float>();

  launch_bwd_contig_partial_specialized<T, 1024, 1, 8192>(
      x, weight, dout, rstd, dx, nullptr, dw_partial, partial_blocks, M, N, stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

template <typename T>
bool try_launch_bwd_contig_half_8192_tpr1024_residual(torch::Tensor const& x_t,
                                                      torch::Tensor const& weight_t,
                                                      torch::Tensor const& dout_t,
                                                      torch::Tensor const& rstd_t,
                                                      torch::Tensor const& dresidual_out_t,
                                                      torch::Tensor const& dx_t,
                                                      torch::Tensor const& dw_t,
                                                      cudaStream_t stream) {
  static_assert(sizeof(T) == 2);
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 8192) {
    return false;
  }

  int const partial_blocks = fast_bwd_partial_blocks(N, x_t.get_device());
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = reinterpret_cast<T const*>(dout_t.const_data_ptr());
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = reinterpret_cast<T*>(dx_t.data_ptr());
  auto const* dresidual_out = reinterpret_cast<T const*>(dresidual_out_t.const_data_ptr());
  auto* dw_partial = partial.data_ptr<float>();

  launch_bwd_contig_partial_specialized<T, 1024, 1, 8192, true>(
      x, weight, dout, rstd, dx, dresidual_out, dw_partial, partial_blocks, M, N, stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}


template <typename T>
bool try_launch_bwd_contig_half_65536_cluster(torch::Tensor const& x_t,
                                              torch::Tensor const& weight_t,
                                              torch::Tensor const& dout_t,
                                              torch::Tensor const& rstd_t,
                                              torch::Tensor const& dx_t,
                                              torch::Tensor const& dw_t,
                                              cudaStream_t stream) {
  constexpr int kClusterN = 2;
  constexpr int kThreadsPerRow = 512;
  constexpr int kMaxVecs = 8;
  constexpr int kDwVecs = 4;
  constexpr int kSharedDwVecs = 4;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 65536) {
    return false;
  }
  int const partial_blocks = sm_count_for_device(x_t.get_device());
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = reinterpret_cast<T const*>(dout_t.const_data_ptr());
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = reinterpret_cast<T*>(dx_t.data_ptr());
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  T const* dresidual_out = nullptr;
  int64_t N_arg = N;
  void* args[] = {
      const_cast<T**>(&x), const_cast<float**>(&weight), const_cast<T**>(&dout),
      const_cast<float**>(&rstd), &dx, const_cast<T**>(&dresidual_out), &dw_partial,
      &M_arg, &N_arg,
  };
  dim3 grid(static_cast<unsigned>(partial_blocks * kClusterN), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = 2 * size_t(kThreadsPerRow) * size_t(kMaxVecs) * sizeof(uint4) +
                size_t(kSharedDwVecs) * size_t(Vec128<T>::kElements) *
                    size_t(kThreadsPerRow) * sizeof(float) +
                (32 + kClusterN) * sizeof(float) + 256;
  auto kernel = rmsnorm_bwd_fullrow_dx_only_kernel<T, kThreadsPerRow, kMaxVecs, 65536,
                                                    kDwVecs, kSharedDwVecs, false, kClusterN>;
  launch_kernel_cluster(reinterpret_cast<void const*>(kernel), grid, block, args, smem,
                        dim3(kClusterN, 1, 1), stream);
  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

template <typename T,
          bool HasDresidualOut = false,
          int kDxDwVecs = 2,
          int kDxSharedDwVecs = 2,
          int kDwThreadsPerRow = 1024,
          int kDwMaxVecs = 2,
          int kDxThreadsPerRow = 512>
bool try_launch_bwd_contig_half_32768_fullrow_dx(torch::Tensor const& x_t,
                                                 torch::Tensor const& weight_t,
                                                 torch::Tensor const& dout_t,
                                                 torch::Tensor const& rstd_t,
                                                 torch::Tensor const& dx_t,
                                                 torch::Tensor const& dw_t,
                                                 cudaStream_t stream,
                                                 T const* dresidual_out = nullptr) {
  static_assert(sizeof(T) == 2);
  constexpr int kDxMaxVecs = 4096 / kDxThreadsPerRow;
  static_assert(kDxThreadsPerRow * kDxMaxVecs == 4096);
  // The full-row dx pass already stages x/dout for the whole row. Accumulate
  // the first 8K dw columns in registers and the next 8K in shared memory,
  // leaving one 16K split-DW tail. A 1024x2 tail keeps fewer dw accumulators
  // per thread than 512x4 and measured faster on B200.
  constexpr int kVec = Vec128<T>::kElements;
  constexpr int64_t kSegmentCols = int64_t(kDwThreadsPerRow) * kVec * kDwMaxVecs;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 32768) {
    return false;
  }

  int partial_blocks = fast_bwd_partial_blocks(N, x_t.get_device());
  int64_t const slices = N / kSegmentCols;
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));

  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = reinterpret_cast<T const*>(dout_t.const_data_ptr());
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = reinterpret_cast<T*>(dx_t.data_ptr());
  auto* dw_partial = partial.data_ptr<float>();

  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 dx_grid(static_cast<unsigned>(partial_blocks), 1, 1);
  dim3 dx_block(kDxThreadsPerRow, 1, 1);
  size_t dx_smem =
      2 * size_t(kDxThreadsPerRow) * size_t(kDxMaxVecs) * sizeof(uint4) +
      size_t(kDxSharedDwVecs) * size_t(kVec) * size_t(kDxThreadsPerRow) * sizeof(float) +
      32 * sizeof(float) + 256;
  void* dx_args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      const_cast<T**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<T**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  static_assert(((kDxDwVecs + kDxSharedDwVecs) * kDxThreadsPerRow) %
                    (kDwThreadsPerRow * kDwMaxVecs) ==
                0);
  int64_t dw_slice_offset =
      (kDxDwVecs + kDxSharedDwVecs) * kDxThreadsPerRow / (kDwThreadsPerRow * kDwMaxVecs);
  int64_t dw_slices = slices - dw_slice_offset;
  dim3 dw_grid(static_cast<unsigned>(partial_blocks), static_cast<unsigned>(dw_slices), 1);
  dim3 dw_block(kDwThreadsPerRow, 1, 1);
  int64_t segment_arg = kSegmentCols;
  void* dw_args[] = {
      const_cast<T**>(&x),
      const_cast<T**>(&dout),
      const_cast<float**>(&rstd),
      &dw_partial,
      &M_arg,
      &N_arg,
      &segment_arg,
      &dw_slice_offset,
  };

  auto dx_kernel =
      rmsnorm_bwd_fullrow_dx_only_kernel<T, kDxThreadsPerRow, kDxMaxVecs, 32768,
                                         kDxDwVecs, kDxSharedDwVecs, HasDresidualOut>;
  launch_kernel(reinterpret_cast<void const*>(dx_kernel), dx_grid, dx_block, dx_args, dx_smem,
                stream);
  if (dw_slices > 0) {
    auto dw_kernel = rmsnorm_bwd_split_dw_only_kernel<T, kDwThreadsPerRow, kDwMaxVecs>;
    launch_kernel(reinterpret_cast<void const*>(dw_kernel), dw_grid, dw_block, dw_args, 0,
                  stream);
  }

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

template <typename T, bool HasDresidualOut = false>
bool try_launch_bwd_contig_half_32768_fullrow_dx_tuned(
    torch::Tensor const& x_t,
    torch::Tensor const& weight_t,
    torch::Tensor const& dout_t,
    torch::Tensor const& rstd_t,
    torch::Tensor const& dx_t,
    torch::Tensor const& dw_t,
    cudaStream_t stream,
    T const* dresidual_out = nullptr) {
  return try_launch_bwd_contig_half_32768_fullrow_dx<T, HasDresidualOut, 4, 4, 1024, 2>(
      x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream, dresidual_out);
}

bool try_launch_bwd_contig_fp32_8192_cluster_reg(torch::Tensor const& x_t,
                                                 torch::Tensor const& weight_t,
                                                 torch::Tensor const& dout_t,
                                                 torch::Tensor const& rstd_t,
                                                 torch::Tensor const& dx_t,
                                                 torch::Tensor const& dw_t,
                                                 cudaStream_t stream) {
  constexpr int kClusterN = 2;
  constexpr int kThreadsPerRow = 256;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 8192) {
    return false;
  }

  int partial_blocks = fast_bwd_partial_blocks(N, x_t.get_device());
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));

  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = dout_t.const_data_ptr<float>();
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = dx_t.data_ptr<float>();
  float const* dresidual_out = nullptr;
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks * kClusterN), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = 32 * sizeof(float) + 16 * sizeof(float) + 256;
  void* kernel_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<float**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  auto kernel =
      rmsnorm_bwd_contig_fp32_cluster_reg_kernel<kClusterN, kThreadsPerRow, 4, false, false,
                                                 false, true, true>;
  launch_kernel_cluster(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem,
                        dim3(kClusterN, 1, 1), stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

bool try_launch_bwd_contig_fp32_32768_cluster_reg(torch::Tensor const& x_t,
                                                  torch::Tensor const& weight_t,
                                                  torch::Tensor const& dout_t,
                                                  torch::Tensor const& rstd_t,
                                                  torch::Tensor const& dx_t,
                                                  torch::Tensor const& dw_t,
                                                  cudaStream_t stream) {
  constexpr int kClusterN = 8;
  constexpr int kThreadsPerRow = 256;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 32768 && N != 65536) {
    return false;
  }

  int partial_blocks =
      N == 32768 ? std::max(sm_count_for_device(x_t.get_device()) * 2, 1)
                 : std::max((sm_count_for_device(x_t.get_device()) * 13) / 16, 1);
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));

  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = dout_t.const_data_ptr<float>();
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = dx_t.data_ptr<float>();
  float const* dresidual_out = nullptr;
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 block(kThreadsPerRow, 1, 1);
  bool const use_double_buffered_sums = N == 65536;
  size_t const cluster_sum_values =
      N == 32768 ? (kThreadsPerRow / 32) * kClusterN
                 : (use_double_buffered_sums ? 32 : kClusterN);
  size_t smem = 32 * sizeof(float) + cluster_sum_values * sizeof(float) +
                (N == 32768 ? sizeof(uint64_t) : 0) + 256;
  void* kernel_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<float**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  if (N == 32768) {
    dim3 grid(static_cast<unsigned>(partial_blocks * kClusterN), 1, 1);
    auto kernel =
        rmsnorm_bwd_contig_fp32_cluster_reg_kernel<kClusterN, kThreadsPerRow, 4, false, false,
                                                   false, true, false, 1, false, false, true, true>;
    launch_kernel_cluster(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem,
                          dim3(kClusterN, 1, 1), stream);
  } else {
    dim3 grid(static_cast<unsigned>(partial_blocks * kClusterN), 1, 1);
    if (use_double_buffered_sums) {
      auto kernel =
          rmsnorm_bwd_contig_fp32_cluster_reg_kernel<kClusterN, kThreadsPerRow, 8, true, true,
                                                     true>;
      launch_kernel_cluster(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem,
                            dim3(kClusterN, 1, 1), stream);
    } else {
      auto kernel =
          rmsnorm_bwd_contig_fp32_cluster_reg_kernel<kClusterN, kThreadsPerRow, 8, true>;
      launch_kernel_cluster(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem,
                            dim3(kClusterN, 1, 1), stream);
    }
  }

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

bool try_launch_bwd_contig_fp32_32768_cluster_reg_residual(
    torch::Tensor const& x_t,
    torch::Tensor const& weight_t,
    torch::Tensor const& dout_t,
    torch::Tensor const& rstd_t,
    torch::Tensor const& dresidual_out_t,
    torch::Tensor const& dx_t,
    torch::Tensor const& dw_t,
    cudaStream_t stream) {
  constexpr int kClusterN = 8;
  constexpr int kThreadsPerRow = 256;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N != 32768) {
    return false;
  }

  int const partial_blocks = std::max(sm_count_for_device(x_t.get_device()) * 2, 1);
  constexpr int kMaxVecs = 4;
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));

  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = dout_t.const_data_ptr<float>();
  auto const* rstd = rstd_t.data_ptr<float>();
  auto const* dresidual_out = dresidual_out_t.const_data_ptr<float>();
  auto* dx = dx_t.data_ptr<float>();
  auto* dw_partial = partial.data_ptr<float>();
  int64_t M_arg = M;
  int64_t N_arg = N;
  dim3 grid(static_cast<unsigned>(partial_blocks * kClusterN), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = 32 * sizeof(float) + (kThreadsPerRow / 32) * kClusterN * sizeof(float) +
                sizeof(uint64_t) +
                2 * size_t(kThreadsPerRow) * size_t(kMaxVecs) * sizeof(uint4) + 256;
  void* kernel_args[] = {
      const_cast<float**>(&x),
      const_cast<float**>(&weight),
      const_cast<float**>(&dout),
      const_cast<float**>(&rstd),
      &dx,
      const_cast<float**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
  };
  auto kernel =
      rmsnorm_bwd_contig_fp32_cluster_reg_kernel<kClusterN, kThreadsPerRow, kMaxVecs, false,
                                                 false, false, true, false, 2, true, true, true,
                                                 true>;
  launch_kernel_cluster(reinterpret_cast<void const*>(kernel), grid, block, kernel_args, smem,
                        dim3(kClusterN, 1, 1), stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);
  return true;
}

template <typename T, int kMaxVecs, bool HasDresidualOut = false>
bool try_launch_bwd_contig_split_impl(torch::Tensor const& x_t,
                                      torch::Tensor const& weight_t,
                                      torch::Tensor const& dout_t,
                                      torch::Tensor const& rstd_t,
                                      torch::Tensor const& dx_t,
                                      torch::Tensor const& dw_t,
                                      cudaStream_t stream,
                                      T const* dresidual_out = nullptr) {
  constexpr int kThreadsPerRow = 256;
  constexpr int64_t kSegmentCols = int64_t(kThreadsPerRow) * Vec128<T>::kElements * kMaxVecs;
  int64_t const M = x_t.size(0);
  int64_t const N = x_t.size(1);
  if (N < 32768 || N % kSegmentCols != 0) {
    return false;
  }
  int64_t const slices = N / kSegmentCols;
  if (slices < 2 || slices > 64) {
    return false;
  }

  int partial_blocks = fast_bwd_partial_blocks(N, x_t.get_device());
  if constexpr (sizeof(T) == 4) {
    if (N >= 131072) {
      partial_blocks = std::max(sm_count_for_device(x_t.get_device()) / 4, 1);
    }
  }
  auto row_dot_partial = torch::empty({M, slices}, x_t.options().dtype(torch::kFloat32));
  auto partial = torch::empty({partial_blocks, N}, x_t.options().dtype(torch::kFloat32));
  bool const pre_reduce_row_dot = slices >= 4;
  auto row_dot_total =
      pre_reduce_row_dot ? torch::empty({M}, x_t.options().dtype(torch::kFloat32))
                         : torch::Tensor();

  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.data_ptr<float>();
  auto const* dout = reinterpret_cast<T const*>(dout_t.const_data_ptr());
  auto const* rstd = rstd_t.data_ptr<float>();
  auto* dx = reinterpret_cast<T*>(dx_t.data_ptr());
  auto* row_dot = row_dot_partial.data_ptr<float>();
  float const* row_dot_total_ptr =
      pre_reduce_row_dot ? row_dot_total.data_ptr<float>() : nullptr;
  auto* row_dot_total_write = pre_reduce_row_dot ? row_dot_total.data_ptr<float>() : nullptr;
  auto* dw_partial = partial.data_ptr<float>();

  dim3 grid(static_cast<unsigned>(partial_blocks), static_cast<unsigned>(slices), 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = smem_bytes(kThreadsPerRow, kThreadsPerRow);

  int64_t M_arg = M;
  int64_t N_arg = N;
  int64_t slices_arg = slices;
  int64_t segment_arg = kSegmentCols;
  void* rowdot_args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      const_cast<T**>(&dout),
      const_cast<float**>(&rstd),
      &row_dot,
      &dw_partial,
      &M_arg,
      &N_arg,
      &slices_arg,
      &segment_arg,
  };
  void const* rowdot_kernel =
      reinterpret_cast<void const*>(
          rmsnorm_bwd_split_rowdot_kernel<T, kThreadsPerRow, kMaxVecs, true, true>);
  launch_kernel(rowdot_kernel, grid, block, rowdot_args, smem, stream);

  float* dw = dw_t.data_ptr<float>();
  launch_reduce_dw_partial(dw_partial, dw, partial_blocks, N, stream, false);

  if (pre_reduce_row_dot) {
    dim3 rowdot_reduce_grid(static_cast<unsigned>((M + 255) / 256), 1, 1);
    dim3 rowdot_reduce_block(256, 1, 1);
    void* rowdot_reduce_args[] = {
        &row_dot,
        &row_dot_total_write,
        &M_arg,
        &slices_arg,
    };
    auto rowdot_reduce_kernel = rmsnorm_reduce_rowdot_partial_kernel;
    launch_kernel(reinterpret_cast<void const*>(rowdot_reduce_kernel), rowdot_reduce_grid,
                  rowdot_reduce_block, rowdot_reduce_args, 0, stream);
  }

  void* partial_args[] = {
      const_cast<T**>(&x),
      const_cast<float**>(&weight),
      const_cast<T**>(&dout),
      const_cast<float**>(&rstd),
      &row_dot,
      const_cast<float**>(&row_dot_total_ptr),
      &dx,
      const_cast<T**>(&dresidual_out),
      &dw_partial,
      &M_arg,
      &N_arg,
      &slices_arg,
      &segment_arg,
  };
  if constexpr (sizeof(T) == 2 && !HasDresidualOut) {
    if (N == 65536) {
      auto partial_kernel =
          rmsnorm_bwd_split_partial_kernel<T, kThreadsPerRow, kMaxVecs, false, false, true>;
      launch_kernel(reinterpret_cast<void const*>(partial_kernel), grid, block, partial_args, 0,
                    stream);
      return true;
    }
  }
  auto partial_kernel =
      rmsnorm_bwd_split_partial_kernel<T, kThreadsPerRow, kMaxVecs, false, HasDresidualOut>;
  launch_kernel(reinterpret_cast<void const*>(partial_kernel), grid, block, partial_args, 0,
                stream);
  return true;
}

template <typename T>
bool try_launch_bwd_contig_split(torch::Tensor const& x_t,
                                 torch::Tensor const& weight_t,
                                 torch::Tensor const& dout_t,
                                 torch::Tensor const& rstd_t,
                                 torch::Tensor const& dx_t,
                                 torch::Tensor const& dw_t,
                                 cudaStream_t stream) {
  if constexpr (sizeof(T) == 4) {
    int64_t const N = x_t.size(1);
    if (N >= 65536 && N <= 131072) {
      return try_launch_bwd_contig_split_impl<T, 8>(
          x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
    }
  }
  return try_launch_bwd_contig_split_impl<T, 4>(
      x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
}

template <typename T>
bool try_launch_bwd_contig_split_residual(torch::Tensor const& x_t,
                                          torch::Tensor const& weight_t,
                                          torch::Tensor const& dout_t,
                                          torch::Tensor const& rstd_t,
                                          torch::Tensor const& dresidual_out_t,
                                          torch::Tensor const& dx_t,
                                          torch::Tensor const& dw_t,
                                          cudaStream_t stream) {
  auto const* dresidual_out = reinterpret_cast<T const*>(dresidual_out_t.const_data_ptr());
  if constexpr (sizeof(T) == 4) {
    int64_t const N = x_t.size(1);
    if (N >= 65536 && N <= 131072) {
      return try_launch_bwd_contig_split_impl<T, 8, true>(
          x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream, dresidual_out);
    }
  }
  return try_launch_bwd_contig_split_impl<T, 4, true>(
      x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream, dresidual_out);
}

template <int ThreadsPerRow, int NumThreads>
void launch_bwd_specialized(Tensor3 x,
                            Affine weight,
                            Tensor3 dout,
                            Tensor2 rstd,
                            Tensor3 dresidual_out,
                            Tensor3 dx,
                            float* dw,
                            float* db,
                            Tensor3 dresidual,
                            int64_t rows_m,
                            int64_t heads,
                            int64_t N,
                            cudaStream_t stream) {
  constexpr int kRowsPerBlock = NumThreads / ThreadsPerRow;
  dim3 grid(static_cast<unsigned>((rows_m * heads + kRowsPerBlock - 1) / kRowsPerBlock), 1, 1);
  dim3 block(NumThreads, 1, 1);
  size_t smem = smem_bytes(NumThreads, ThreadsPerRow);
  auto kernel = rmsnorm_bwd_kernel<ThreadsPerRow, NumThreads>;
  void* args[] = {
      &x, &weight, &dout, &rstd, &dresidual_out, &dx, &dw, &db, &dresidual,
      &rows_m, &heads, &N,
  };
  launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, args, smem, stream);
}

template <int NumThreads>
void dispatch_bwd_threads(int threads_per_row,
                          Tensor3 x,
                          Affine weight,
                          Tensor3 dout,
                          Tensor2 rstd,
                          Tensor3 dresidual_out,
                          Tensor3 dx,
                          float* dw,
                          float* db,
                          Tensor3 dresidual,
                          int64_t rows_m,
                          int64_t heads,
                          int64_t N,
                          cudaStream_t stream) {
  switch (threads_per_row) {
    case 8:
      launch_bwd_specialized<8, NumThreads>(x, weight, dout, rstd, dresidual_out, dx, dw, db,
                                            dresidual, rows_m, heads, N, stream);
      break;
    case 16:
      launch_bwd_specialized<16, NumThreads>(x, weight, dout, rstd, dresidual_out, dx, dw, db,
                                             dresidual, rows_m, heads, N, stream);
      break;
    case 32:
      launch_bwd_specialized<32, NumThreads>(x, weight, dout, rstd, dresidual_out, dx, dw, db,
                                             dresidual, rows_m, heads, N, stream);
      break;
    case 64:
      launch_bwd_specialized<64, NumThreads>(x, weight, dout, rstd, dresidual_out, dx, dw, db,
                                             dresidual, rows_m, heads, N, stream);
      break;
    case 128:
      launch_bwd_specialized<128, NumThreads>(x, weight, dout, rstd, dresidual_out, dx, dw, db,
                                              dresidual, rows_m, heads, N, stream);
      break;
    case 256:
      if constexpr (NumThreads >= 256) {
        launch_bwd_specialized<256, NumThreads>(x, weight, dout, rstd, dresidual_out, dx, dw, db,
                                                dresidual, rows_m, heads, N, stream);
      } else {
        TORCH_CHECK(false, "threads_per_row exceeds block size");
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported threads_per_row=", threads_per_row);
  }
}

void rmsnorm_fwd(torch::Tensor const& x_t,
                 torch::Tensor const& weight_t,
                 torch::Tensor const& bias_t,
                 torch::Tensor const& residual_t,
                 torch::Tensor const& out_t,
                 torch::Tensor const& residual_out_t,
                 torch::Tensor const& rstd_t,
                 double eps) {
  TORCH_CHECK(x_t.is_cuda(), "x must be a CUDA tensor");
  for (auto const& named : {std::pair{"weight", &weight_t}, std::pair{"bias", &bias_t},
                            std::pair{"residual", &residual_t}, std::pair{"out", &out_t},
                            std::pair{"residual_out", &residual_out_t},
                            std::pair{"rstd", &rstd_t}}) {
    TORCH_CHECK(named.second->is_cuda(), named.first, " must be a CUDA tensor");
    TORCH_CHECK(named.second->device() == x_t.device(), named.first, " must be on ", x_t.device());
  }
  TORCH_CHECK(x_t.dim() == 2 || x_t.dim() == 3, "x must be 2D or 3D");
  TORCH_CHECK(out_t.sizes() == x_t.sizes(), "out shape must match x");
  TORCH_CHECK(x_t.stride(x_t.dim() - 1) == 1, "x last dimension must be contiguous");
  TORCH_CHECK(out_t.stride(out_t.dim() - 1) == 1, "out last dimension must be contiguous");
  if (x_t.numel() == 0) {
    return;
  }

  int64_t const rows_m = x_t.size(0);
  int64_t const heads = x_t.dim() == 3 ? x_t.size(1) : 1;
  int64_t const N = x_t.size(x_t.dim() - 1);
  TORCH_CHECK(N > 0, "N must be positive");

  Tensor3 x = make_tensor3(x_t, false);
  Tensor3 residual = make_tensor3(residual_t, true);
  Tensor3 out = make_tensor3(out_t, false);
  Tensor3 residual_out = make_tensor3(residual_out_t, true);
  Tensor2 rstd = make_tensor2(rstd_t, x_t.dim(), true);
  Affine weight = make_affine(weight_t, heads, N);
  Affine bias = make_affine(bias_t, heads, N);

  if (residual.dtype != DType::kNone) {
    TORCH_CHECK(residual_t.sizes() == x_t.sizes(), "residual shape must match x");
  }
  if (residual_out.dtype != DType::kNone) {
    TORCH_CHECK(residual_out_t.sizes() == x_t.sizes(), "residual_out shape must match x");
  }

  int threads_per_row = threads_per_row_fwd(N);
  int num_threads = N <= 16384 ? 128 : 256;
  if (threads_per_row > num_threads) {
    num_threads = threads_per_row;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bool const has_residual = residual_t.numel() != 0;
  bool const has_rstd_out = rstd_t.numel() != 0;
  bool const fast_contig =
      x_t.dim() == 2 && x_t.is_contiguous() && out_t.is_contiguous() &&
      out_t.scalar_type() == x_t.scalar_type() && weight_t.numel() != 0 &&
      weight_t.scalar_type() == at::ScalarType::Float && weight_t.dim() == 1 &&
      weight_t.is_contiguous() && bias_t.numel() == 0 &&
      (!has_residual ||
       (residual_t.scalar_type() == x_t.scalar_type() && residual_t.is_contiguous() &&
        residual_out_t.scalar_type() == x_t.scalar_type() && residual_out_t.is_contiguous())) &&
      (has_residual || residual_out_t.numel() == 0) &&
      (!has_rstd_out ||
       (rstd_t.scalar_type() == at::ScalarType::Float && rstd_t.is_contiguous()));
  if (fast_contig) {
    bool launched = false;
    int fast_threads_per_row = threads_per_row;
    int fast_num_threads = num_threads;
    if (N == 2048 &&
        (x_t.scalar_type() == at::ScalarType::Half ||
         x_t.scalar_type() == at::ScalarType::BFloat16)) {
      fast_threads_per_row = 128;
    } else if (x_t.scalar_type() == at::ScalarType::Float) {
      if (N == 256 || N == 512) {
        fast_num_threads = 256;
      } else if (N == 1024) {
        fast_threads_per_row = 128;
        fast_num_threads = 256;
      } else if (N == 2048 || N == 4096 || N == 8192) {
        fast_threads_per_row = 256;
      }
    }
    if (fast_threads_per_row > fast_num_threads) {
      fast_num_threads = fast_threads_per_row;
    }
    bool const prefer_fwd_split =
        !has_residual && !has_rstd_out &&
        ((x_t.scalar_type() == at::ScalarType::Float && N >= 32768) ||
         ((x_t.scalar_type() == at::ScalarType::Half ||
           x_t.scalar_type() == at::ScalarType::BFloat16) &&
          N >= 32768));
    if (prefer_fwd_split) {
      if (x_t.scalar_type() == at::ScalarType::Half) {
        launched = try_launch_fwd_contig_cluster<half>(x_t, weight_t, out_t, eps, stream);
        if (!launched) {
          launched = try_launch_fwd_contig_stream<half>(x_t, weight_t, out_t, eps, stream);
        }
        if (!launched) {
          launched = try_launch_fwd_contig_split<half>(x_t, weight_t, out_t, eps, stream);
        }
      } else if (x_t.scalar_type() == at::ScalarType::BFloat16) {
        launched = try_launch_fwd_contig_cluster<__nv_bfloat16>(
            x_t, weight_t, out_t, eps, stream);
        if (!launched) {
          launched = try_launch_fwd_contig_stream<__nv_bfloat16>(
              x_t, weight_t, out_t, eps, stream);
        }
        if (!launched) {
          launched = try_launch_fwd_contig_split<__nv_bfloat16>(
              x_t, weight_t, out_t, eps, stream);
        }
      } else if (x_t.scalar_type() == at::ScalarType::Float) {
        launched = try_launch_fwd_contig_cluster<float>(x_t, weight_t, out_t, eps, stream);
        if (!launched) {
          launched = try_launch_fwd_contig_stream<float>(x_t, weight_t, out_t, eps, stream);
        }
        if (!launched) {
          launched = try_launch_fwd_contig_split<float>(x_t, weight_t, out_t, eps, stream);
        }
      }
    }
    if (!launched && has_residual && N >= 65536) {
      if (x_t.scalar_type() == at::ScalarType::Half) {
        launched = try_launch_fwd_contig_split_residual<half>(
            x_t, weight_t, residual_t, out_t, residual_out_t, rstd_t, eps, stream);
      } else if (x_t.scalar_type() == at::ScalarType::BFloat16) {
        launched = try_launch_fwd_contig_split_residual<__nv_bfloat16>(
            x_t, weight_t, residual_t, out_t, residual_out_t, rstd_t, eps, stream);
      }
    }
    if (!launched && x_t.scalar_type() == at::ScalarType::Half) {
      launched = try_launch_fwd_contig<half>(x_t, weight_t, bias_t, residual_t, out_t,
                                             residual_out_t, rstd_t, fast_threads_per_row,
                                             fast_num_threads, eps, stream);
    } else if (!launched && x_t.scalar_type() == at::ScalarType::BFloat16) {
      launched = try_launch_fwd_contig<__nv_bfloat16>(
          x_t, weight_t, bias_t, residual_t, out_t, residual_out_t, rstd_t, fast_threads_per_row,
          fast_num_threads, eps, stream);
    } else if (!launched && x_t.scalar_type() == at::ScalarType::Float) {
      if (!has_residual && !has_rstd_out) {
        launched = try_launch_fwd_contig_fp32_preload_w(x_t, weight_t, out_t, eps, stream);
        if (launched) {
          CUDA_KERNEL_CHECK();
          return;
        }
      }
      launched = try_launch_fwd_contig<float>(x_t, weight_t, bias_t, residual_t, out_t,
                                              residual_out_t, rstd_t, fast_threads_per_row,
                                              fast_num_threads, eps, stream);
    }
    if (!launched && !has_residual && !has_rstd_out) {
      if (x_t.scalar_type() == at::ScalarType::Half) {
        launched = try_launch_fwd_contig_stream<half>(x_t, weight_t, out_t, eps, stream);
        if (!launched) {
          launched = try_launch_fwd_contig_split<half>(x_t, weight_t, out_t, eps, stream);
        }
      } else if (x_t.scalar_type() == at::ScalarType::BFloat16) {
        launched = try_launch_fwd_contig_stream<__nv_bfloat16>(
            x_t, weight_t, out_t, eps, stream);
        if (!launched) {
          launched = try_launch_fwd_contig_split<__nv_bfloat16>(
              x_t, weight_t, out_t, eps, stream);
        }
      } else if (x_t.scalar_type() == at::ScalarType::Float) {
        launched = try_launch_fwd_contig_cluster<float>(x_t, weight_t, out_t, eps, stream);
        if (!launched) {
          launched = try_launch_fwd_contig_stream<float>(x_t, weight_t, out_t, eps, stream);
        }
        if (!launched) {
          launched = try_launch_fwd_contig_split<float>(x_t, weight_t, out_t, eps, stream);
        }
      }
    }
    if (launched) {
      CUDA_KERNEL_CHECK();
      return;
    }
  }
  if (num_threads == 128) {
    dispatch_fwd_threads<128>(threads_per_row, x, weight, bias, residual, out, residual_out, rstd,
                              rows_m, heads, N, static_cast<float>(eps), stream);
  } else {
    dispatch_fwd_threads<256>(threads_per_row, x, weight, bias, residual, out, residual_out, rstd,
                              rows_m, heads, N, static_cast<float>(eps), stream);
  }
  CUDA_KERNEL_CHECK();
}

void rmsnorm_bwd(torch::Tensor const& x_t,
                 torch::Tensor const& weight_t,
                 torch::Tensor const& dout_t,
                 torch::Tensor const& rstd_t,
                 torch::Tensor const& dresidual_out_t,
                 torch::Tensor const& dx_t,
                 torch::Tensor const& dw_t,
                 torch::Tensor const& db_t,
                 torch::Tensor const& dresidual_t) {
  TORCH_CHECK(x_t.is_cuda(), "x must be a CUDA tensor");
  for (auto const& named : {std::pair{"weight", &weight_t}, std::pair{"dout", &dout_t},
                            std::pair{"rstd", &rstd_t},
                            std::pair{"dresidual_out", &dresidual_out_t}, std::pair{"dx", &dx_t},
                            std::pair{"dw", &dw_t}, std::pair{"db", &db_t},
                            std::pair{"dresidual", &dresidual_t}}) {
    TORCH_CHECK(named.second->is_cuda(), named.first, " must be a CUDA tensor");
    TORCH_CHECK(named.second->device() == x_t.device(), named.first, " must be on ", x_t.device());
  }
  TORCH_CHECK(x_t.dim() == 2 || x_t.dim() == 3, "x must be 2D or 3D");
  TORCH_CHECK(dout_t.sizes() == x_t.sizes(), "dout shape must match x");
  TORCH_CHECK(dx_t.sizes() == x_t.sizes(), "dx shape must match x");
  TORCH_CHECK(x_t.stride(x_t.dim() - 1) == 1, "x last dimension must be contiguous");
  TORCH_CHECK(dout_t.stride(dout_t.dim() - 1) == 1, "dout last dimension must be contiguous");
  TORCH_CHECK(dx_t.stride(dx_t.dim() - 1) == 1, "dx last dimension must be contiguous");
  if (x_t.numel() == 0) {
    return;
  }

  int64_t const rows_m = x_t.size(0);
  int64_t const heads = x_t.dim() == 3 ? x_t.size(1) : 1;
  int64_t const N = x_t.size(x_t.dim() - 1);
  Tensor3 x = make_tensor3(x_t, false);
  Tensor3 dout = make_tensor3(dout_t, false);
  Tensor3 dresidual_out = make_tensor3(dresidual_out_t, true);
  Tensor3 dx = make_tensor3(dx_t, false);
  Tensor3 dresidual = make_tensor3(dresidual_t, true);
  Tensor2 rstd = make_tensor2(rstd_t, x_t.dim(), false);
  Affine weight = make_affine(weight_t, heads, N);

  if (dresidual_out.dtype != DType::kNone) {
    TORCH_CHECK(dresidual_out_t.sizes() == x_t.sizes(), "dresidual_out shape must match x");
  }
  if (dresidual.dtype != DType::kNone) {
    TORCH_CHECK(dresidual_t.sizes() == x_t.sizes(), "dresidual shape must match x");
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  float* dw = nullptr;
  if (dw_t.numel() != 0) {
    TORCH_CHECK(dw_t.is_cuda() && dw_t.scalar_type() == at::ScalarType::Float,
                "dw accumulator must be a CUDA float32 tensor");
    TORCH_CHECK(dw_t.numel() == heads * N, "dw accumulator shape mismatch");
    dw = dw_t.data_ptr<float>();
  }
  float* db = nullptr;
  if (db_t.numel() != 0) {
    TORCH_CHECK(db_t.is_cuda() && db_t.scalar_type() == at::ScalarType::Float,
                "db accumulator must be a CUDA float32 tensor");
    TORCH_CHECK(db_t.numel() == heads * N, "db accumulator shape mismatch");
    db = db_t.data_ptr<float>();
  }

  int threads_per_row = threads_per_row_bwd(N);
  int num_threads = N <= 4096 ? 128 : 256;
  if (threads_per_row > num_threads) {
    num_threads = threads_per_row;
  }
  bool const fast_contig_bwd =
      x_t.dim() == 2 &&
      (x_t.scalar_type() == at::ScalarType::Half ||
       x_t.scalar_type() == at::ScalarType::BFloat16 ||
       x_t.scalar_type() == at::ScalarType::Float) &&
      x_t.is_contiguous() && dout_t.is_contiguous() && dx_t.is_contiguous() &&
      dout_t.scalar_type() == x_t.scalar_type() && dx_t.scalar_type() == x_t.scalar_type() &&
      weight_t.numel() != 0 && weight_t.scalar_type() == at::ScalarType::Float &&
      weight_t.dim() == 1 && weight_t.is_contiguous() && rstd_t.is_contiguous() &&
      dresidual_out.dtype == DType::kNone && dresidual.dtype == DType::kNone &&
      dw != nullptr && db == nullptr && dw_t.is_contiguous();
  bool const fast_contig_bwd_residual =
      x_t.dim() == 2 &&
      (x_t.scalar_type() == at::ScalarType::Half ||
       x_t.scalar_type() == at::ScalarType::BFloat16 ||
       x_t.scalar_type() == at::ScalarType::Float) &&
      x_t.is_contiguous() && dout_t.is_contiguous() && dx_t.is_contiguous() &&
      dout_t.scalar_type() == x_t.scalar_type() && dx_t.scalar_type() == x_t.scalar_type() &&
      dresidual_out_t.numel() != 0 && dresidual_out_t.is_contiguous() &&
      dresidual_out_t.scalar_type() == x_t.scalar_type() && dresidual.dtype == DType::kNone &&
      weight_t.numel() != 0 && weight_t.scalar_type() == at::ScalarType::Float &&
      weight_t.dim() == 1 && weight_t.is_contiguous() && rstd_t.is_contiguous() &&
      dw != nullptr && db == nullptr && dw_t.is_contiguous();
  if (fast_contig_bwd) {
    int fast_threads_per_row = threads_per_row;
    if (x_t.scalar_type() == at::ScalarType::Half ||
        x_t.scalar_type() == at::ScalarType::BFloat16) {
      if (N == 512) {
        fast_threads_per_row = 32;
      } else if (N == 2048) {
        fast_threads_per_row = 256;
      }
    }
    bool launched = false;
    if (x_t.scalar_type() == at::ScalarType::Float) {
      launched = try_launch_bwd_contig_fp32_16384_smem_dw(
          x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      if (!launched) {
        launched = try_launch_bwd_contig_fp32_8192_cp_async(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        launched = try_launch_bwd_contig_fp32_8192_cluster_reg(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        launched = try_launch_bwd_contig_fp32_32768_cluster_reg(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
    }
    if (x_t.scalar_type() == at::ScalarType::Float && !launched) {
      launched = try_launch_bwd_contig_fp32_32768_wide(
          x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
    }
    if (!launched && x_t.scalar_type() == at::ScalarType::Half) {
      launched = try_launch_bwd_contig_half_cp_async<half>(
          x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      if (!launched) {
        launched = try_launch_bwd_contig_half_8192_tpr1024<half>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        launched = try_launch_bwd_contig_half_32768_fullrow_dx_tuned<half>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        launched = try_launch_bwd_contig_half_65536_cluster<half>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
    } else if (!launched && x_t.scalar_type() == at::ScalarType::BFloat16) {
      launched = try_launch_bwd_contig_half_cp_async<__nv_bfloat16>(
          x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      if (!launched) {
        launched = try_launch_bwd_contig_half_8192_tpr1024<__nv_bfloat16>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        launched = try_launch_bwd_contig_half_32768_fullrow_dx_tuned<__nv_bfloat16>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        launched = try_launch_bwd_contig_half_65536_cluster<__nv_bfloat16>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
    }
    if (!launched && N >= 32768) {
      if (x_t.scalar_type() == at::ScalarType::Half) {
        launched = try_launch_bwd_contig_split<half>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      } else if (x_t.scalar_type() == at::ScalarType::BFloat16) {
        launched = try_launch_bwd_contig_split<__nv_bfloat16>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      } else if (x_t.scalar_type() == at::ScalarType::Float) {
        launched = try_launch_bwd_contig_split<float>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream);
      }
    }
    if (!launched) {
      if (x_t.scalar_type() == at::ScalarType::Half) {
        launched = try_launch_bwd_contig<half>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, fast_threads_per_row, stream);
      } else if (x_t.scalar_type() == at::ScalarType::BFloat16) {
        launched = try_launch_bwd_contig<__nv_bfloat16>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, fast_threads_per_row, stream);
      } else if (x_t.scalar_type() == at::ScalarType::Float) {
        launched = try_launch_bwd_contig<float>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, fast_threads_per_row, stream);
      }
    }
    if (launched) {
      CUDA_KERNEL_CHECK();
      return;
    }
  }
  if (fast_contig_bwd_residual) {
    bool launched = false;
    int fast_threads_per_row = threads_per_row;
    if (x_t.scalar_type() == at::ScalarType::Half ||
        x_t.scalar_type() == at::ScalarType::BFloat16) {
      if (N == 512) {
        fast_threads_per_row = 32;
      } else if (N == 2048) {
        fast_threads_per_row = 256;
      }
    }
    if (x_t.scalar_type() == at::ScalarType::Half) {
      launched = try_launch_bwd_contig_half_8192_tpr1024_residual<half>(
          x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      if (!launched) {
        launched = try_launch_bwd_contig_half_cp_async_residual<half>(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        auto const* dresidual_out =
            reinterpret_cast<half const*>(dresidual_out_t.const_data_ptr());
        launched = try_launch_bwd_contig_half_32768_fullrow_dx_tuned<half, true>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream, dresidual_out);
      }
      if (!launched && N >= 32768) {
        launched = try_launch_bwd_contig_split_residual<half>(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      }
    } else if (x_t.scalar_type() == at::ScalarType::BFloat16) {
      launched = try_launch_bwd_contig_half_8192_tpr1024_residual<__nv_bfloat16>(
          x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      if (!launched) {
        launched = try_launch_bwd_contig_half_cp_async_residual<__nv_bfloat16>(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        auto const* dresidual_out =
            reinterpret_cast<__nv_bfloat16 const*>(dresidual_out_t.const_data_ptr());
        launched = try_launch_bwd_contig_half_32768_fullrow_dx_tuned<__nv_bfloat16, true>(
            x_t, weight_t, dout_t, rstd_t, dx_t, dw_t, stream, dresidual_out);
      }
      if (!launched && N >= 32768) {
        launched = try_launch_bwd_contig_split_residual<__nv_bfloat16>(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      }
    } else if (x_t.scalar_type() == at::ScalarType::Float) {
      launched = try_launch_bwd_contig_fp32_cp_async_residual(
          x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      if (!launched) {
        launched = try_launch_bwd_contig_fp32_8192_cp_async_residual(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        launched = try_launch_bwd_contig_fp32_16384_cluster_reg_residual(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      }
      if (!launched) {
        launched = try_launch_bwd_contig_fp32_32768_cluster_reg_residual(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      }
      if (!launched && N >= 32768) {
        launched = try_launch_bwd_contig_split_residual<float>(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, stream);
      }
    }
    if (!launched) {
      if (x_t.scalar_type() == at::ScalarType::Half) {
        launched = try_launch_bwd_contig_residual<half>(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, fast_threads_per_row,
            stream);
      } else if (x_t.scalar_type() == at::ScalarType::BFloat16) {
        launched = try_launch_bwd_contig_residual<__nv_bfloat16>(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, fast_threads_per_row,
            stream);
      } else if (x_t.scalar_type() == at::ScalarType::Float) {
        launched = try_launch_bwd_contig_residual<float>(
            x_t, weight_t, dout_t, rstd_t, dresidual_out_t, dx_t, dw_t, fast_threads_per_row,
            stream);
      }
    }
    if (launched) {
      CUDA_KERNEL_CHECK();
      return;
    }
  }
  float* no_dw = nullptr;
  float* no_db = nullptr;
  if (num_threads == 128) {
    dispatch_bwd_threads<128>(threads_per_row, x, weight, dout, rstd, dresidual_out, dx, no_dw,
                              no_db, dresidual, rows_m, heads, N, stream);
  } else {
    dispatch_bwd_threads<256>(threads_per_row, x, weight, dout, rstd, dresidual_out, dx, no_dw,
                              no_db, dresidual, rows_m, heads, N, stream);
  }
  if (dw != nullptr || db != nullptr) {
    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(heads), 1);
    dim3 block(256, 1, 1);
    size_t smem = 16 * sizeof(float);
    auto kernel = rmsnorm_affine_grad_kernel;
    int64_t rows_m_arg = rows_m;
    int64_t heads_arg = heads;
    int64_t N_arg = N;
    void* args[] = {&x, &dout, &rstd, &dw, &db, &rows_m_arg, &heads_arg, &N_arg};
    launch_kernel(reinterpret_cast<void const*>(kernel), grid, block, args, smem, stream);
  }
  CUDA_KERNEL_CHECK();
}

void rmsnorm_fwd_fp32_8192_plain(torch::Tensor const& x_t,
                                 torch::Tensor const& weight_t,
                                 torch::Tensor const& out_t,
                                 double eps) {
  auto const* x = x_t.const_data_ptr<float>();
  auto const* weight = weight_t.const_data_ptr<float>();
  auto* out = out_t.data_ptr<float>();
  float eps_arg = static_cast<float>(eps);
  constexpr int kThreadsPerRow = 128;
  constexpr int kMaxVecs = 16;
  dim3 grid(static_cast<unsigned>(x_t.size(0)), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = size_t(kThreadsPerRow) * size_t(kMaxVecs) * sizeof(uint4) +
                size_t(kThreadsPerRow / 32) * sizeof(float) + sizeof(uint64_t);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x_t.get_device());
  rmsnorm_fwd_contig_fp32_smem_async_preload_w_kernel<kThreadsPerRow, kMaxVecs>
      <<<grid, block, smem, stream>>>(x, weight, out, eps_arg);
}

void rmsnorm_fwd_fp16_32768_plain(torch::Tensor const& x_t,
                                  torch::Tensor const& weight_t,
                                  torch::Tensor const& out_t,
                                  double eps) {
  auto const* x = reinterpret_cast<half const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.const_data_ptr<float>();
  auto* out = reinterpret_cast<half*>(out_t.data_ptr());
  int64_t M_arg = x_t.size(0);
  int64_t N_arg = 32768;
  float eps_arg = static_cast<float>(eps);
  constexpr int kClusterN = 2;
  constexpr int kThreadsPerRow = 256;
  constexpr int kMaxVecs = 8;
  dim3 grid(static_cast<unsigned>(M_arg * kClusterN), 1, 1);
  dim3 block(kThreadsPerRow, 1, 1);
  size_t smem = 32 * sizeof(float) + kClusterN * sizeof(float) + 256;
  auto kernel = rmsnorm_fwd_cluster_kernel<half, kClusterN, kThreadsPerRow, kMaxVecs,
                                           false, 6, 0>;
  void* args[] = {const_cast<half**>(&x), const_cast<float**>(&weight), &out,
                  &M_arg, &N_arg, &eps_arg};
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = kClusterN;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = at::cuda::getCurrentCUDAStream(x_t.get_device());
  config.attrs = &attr;
  config.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(kernel), args) ==
                  cudaSuccess,
              "CUDA fp16 32768 forward launch failed");
}

template <typename T, int ThreadsPerRow, int NumThreads, int MaxVecs, bool PreloadWeight = false>
void launch_fwd_residual_plain_direct(torch::Tensor const& x_t,
                                      torch::Tensor const& weight_t,
                                      torch::Tensor const& residual_t,
                                      torch::Tensor const& out_t,
                                      torch::Tensor const& residual_out_t,
                                      double eps) {
  auto const* x = reinterpret_cast<T const*>(x_t.const_data_ptr());
  auto const* weight = weight_t.const_data_ptr<float>();
  auto const* residual = reinterpret_cast<T const*>(residual_t.const_data_ptr());
  auto* out = reinterpret_cast<T*>(out_t.data_ptr());
  auto* residual_out = reinterpret_cast<T*>(residual_out_t.data_ptr());
  float* rstd = nullptr;
  float const* bias = nullptr;
  int64_t M_arg = x_t.size(0);
  int64_t N_arg = x_t.size(1);
  float eps_arg = static_cast<float>(eps);
  constexpr int kRowsPerBlock = NumThreads / ThreadsPerRow;
  dim3 grid(static_cast<unsigned>((M_arg + kRowsPerBlock - 1) / kRowsPerBlock), 1, 1);
  dim3 block(NumThreads, 1, 1);
  size_t smem = smem_bytes(NumThreads, ThreadsPerRow);
  rmsnorm_fwd_contig_kernel<T, ThreadsPerRow, NumThreads, MaxVecs, true, false, PreloadWeight>
      <<<grid, block, smem, at::cuda::getCurrentCUDAStream(x_t.get_device())>>>(
          x, weight, bias, residual, out, residual_out, rstd, M_arg, N_arg, eps_arg);
}

void rmsnorm_fwd_residual_plain(torch::Tensor const& x_t,
                                torch::Tensor const& weight_t,
                                torch::Tensor const& residual_t,
                                torch::Tensor const& out_t,
                                torch::Tensor const& residual_out_t,
                                double eps) {
  int64_t const N = x_t.size(1);
  if (x_t.scalar_type() == at::ScalarType::Half) {
    if (N == 1024) {
      return launch_fwd_residual_plain_direct<half, 128, 128, 1, true>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
    if (N == 2048) {
      return launch_fwd_residual_plain_direct<half, 128, 128, 2, true>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
    if (N == 4096) {
      return launch_fwd_residual_plain_direct<half, 256, 256, 2>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
    if (N == 8192) {
      return launch_fwd_residual_plain_direct<half, 256, 256, 4>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
  } else if (x_t.scalar_type() == at::ScalarType::BFloat16) {
    if (N == 1024) {
      return launch_fwd_residual_plain_direct<__nv_bfloat16, 64, 128, 2, true>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
    if (N == 2048) {
      return launch_fwd_residual_plain_direct<__nv_bfloat16, 128, 128, 2, true>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
    if (N == 4096) {
      return launch_fwd_residual_plain_direct<__nv_bfloat16, 128, 256, 4, true>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
    if (N == 8192) {
      return launch_fwd_residual_plain_direct<__nv_bfloat16, 256, 256, 4, true>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
  } else if (x_t.scalar_type() == at::ScalarType::Float) {
    if (N == 512) {
      return launch_fwd_residual_plain_direct<float, 32, 256, 4, true>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
    if (N == 2048) {
      return launch_fwd_residual_plain_direct<float, 256, 256, 2, true>(
          x_t, weight_t, residual_t, out_t, residual_out_t, eps);
    }
  }
  TORCH_CHECK(false, "unsupported direct residual forward shape");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_fwd", &rmsnorm_fwd, "CUDA/PTX RMSNorm forward");
  m.def("rmsnorm_fwd_fp32_8192_plain", &rmsnorm_fwd_fp32_8192_plain,
        "CUDA/PTX RMSNorm fp32 8192 plain forward");
  m.def("rmsnorm_fwd_fp16_32768_plain", &rmsnorm_fwd_fp16_32768_plain,
        "CUDA/PTX RMSNorm fp16 32768 plain forward");
  m.def("rmsnorm_fwd_residual_plain", &rmsnorm_fwd_residual_plain,
        "CUDA/PTX RMSNorm contiguous residual forward");
  m.def("rmsnorm_bwd", &rmsnorm_bwd, "CUDA/PTX RMSNorm backward");
}
