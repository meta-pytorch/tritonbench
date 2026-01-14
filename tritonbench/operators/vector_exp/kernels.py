import torch
import triton
import triton.language as tl
from tritonbench.kernels.profile import time


@triton.jit
def triton_exp_kernel(
    x_ptr,  # *Pointer* to input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
    profile_mem=None,  # *Pointer* to profile_mem.
):
    if profile_mem is not None:
        start = time()
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.exp(x)
    # Write exp(x) back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

    if profile_mem is not None:
        end = time()
        tl.store(profile_mem + pid, end - start)


@triton.jit
def triton_exp_backward_kernel(
    grad_output_ptr,  # *Pointer* to grad_output vector.
    output_ptr,  # *Pointer* to forward pass output vector (exp(x)).
    grad_input_ptr,  # *Pointer* to grad_input vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    profile_mem=None,  # *Pointer* to profile_mem.
):
    if profile_mem is not None:
        start = time()

    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    # This program will process inputs that are offset from the initial data.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements

    # Load grad_output and output from DRAM
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    output = tl.load(output_ptr + offsets, mask=mask)

    # Compute grad_input = grad_output * output (since d/dx(exp(x)) = exp(x))
    grad_input = grad_output * output

    # Write grad_input back to DRAM.
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)

    if profile_mem is not None:
        end = time()
        tl.store(profile_mem + pid, end - start)


class TritonExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, block_size: int = 1024, profile_mem: torch.Tensor = None
    ):
        # Allocate output tensor
        output = torch.empty_like(x)
        n_elements = output.numel()

        # Launch grid - number of blocks needed
        grid = lambda meta: (triton.cdiv(n_elements, block_size),)

        # Launch forward kernel
        triton_exp_kernel[grid](
            x, output, n_elements, BLOCK_SIZE=block_size, profile_mem=profile_mem
        )

        # Save output for backward pass
        ctx.save_for_backward(output)
        ctx.block_size = block_size
        ctx.profile_mem = profile_mem

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve saved tensors
        (output,) = ctx.saved_tensors

        # Allocate grad_input tensor
        grad_input = torch.empty_like(grad_output)
        n_elements = grad_output.numel()

        # Launch grid - number of blocks needed
        grid = lambda meta: (triton.cdiv(n_elements, ctx.block_size),)

        # Launch backward kernel
        triton_exp_backward_kernel[grid](
            grad_output,
            output,
            grad_input,
            n_elements,
            BLOCK_SIZE=ctx.block_size,
            profile_mem=ctx.profile_mem,
        )

        # Return gradients (None for block_size and profile_mem as they don't need gradients)
        return grad_input, None, None
