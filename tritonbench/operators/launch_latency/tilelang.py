import tilelang
import tilelang.language as T

@tilelang.jit()
def tilelang_nop_kernel():

    @T.prim_func
    def nop_kernel():
        with T.Kernel(1) as bx:
            pass
    return nop_kernel


@tilelang.jit()
def tilelang_nop_with_args_kernel():
    @T.prim_func
    def nop_with_args_kernel(
        t1: T.Tensor((1,), T.float32),
        t2: T.Tensor((1,), T.float32),
        t3: T.Tensor((1,), T.float32),
        t4: T.Tensor((1,), T.float32),
        t5: T.Tensor((1,), T.float32),
        i1: T.int32,
        i2: T.int32,
        i3: T.int32,
        i4: T.int32,
        i5: T.int32,
        i6: T.int32,
        i7: T.int32,
        i8: T.int32,
        i9: T.int32,
        c1: T.int32,
        c2: T.int32,
        c3: T.int32,
        c4: T.int32,
        c5: T.int32,
    ):
        with T.Kernel(1) as bx:
            pass
    return nop_with_args_kernel
