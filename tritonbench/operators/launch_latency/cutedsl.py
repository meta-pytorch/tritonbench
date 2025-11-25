import cutlass
import cutlass.cute as cute

@cute.kernel
def nop_kernel():
    pass

@cute.jit
def cutedsl_nop_kernel():
    nop_kernel()

@cute.kernel
def nop_with_args_kernel(
    t1: cute.Tensor,
    t2: cute.Tensor,
    t3: cute.Tensor,
    t4: cute.Tensor,
    t5: cute.Tensor,
    i1: cutlass.Int32,
    i2: cutlass.Int32,
    i3: cutlass.Int32,
    i4: cutlass.Int32,
    i5: cutlass.Int32,
    i6: cutlass.Int32,
    i7: cutlass.Int32,
    i8: cutlass.Int32,
    i9: cutlass.Int32,
    c1: cutlass.Constexpr,
    c2: cutlass.Constexpr,
    c3: cutlass.Constexpr,
    c4: cutlass.Constexpr,
    c5: cutlass.Constexpr,
):
    pass

@cute.jit
def cutedsl_nop_with_args_kernel(*args):
    nop_with_args_kernel(*args)
