import triton

@triton.jit
def example_triton_kernel():
    pass


if __name__ == "__main__":
    print("Running Triton kernel...")
    example_triton_kernel[1, 1]()
    print("Done!")
