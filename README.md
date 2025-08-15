# TritonBench

TritonBench is a collection of PyTorch operators used to evaluation the performance of [Triton](https://github.com/triton-lang/triton),
and its integration with PyTorch.


## Installation

The benchmark suite should be self-contained of its dependencies. To install, follow the steps below.


Step 1: clone the repository and checkout all submodules

```
$ git clone https://github.com/meta-pytorch/tritonbench.git
$ git submodule update --init --recursive
```

Step 2: run install.py

```
$ python install.py
```

By default, it will install the latest PyTorch nightly release and use the Triton version bundled with it.

## Basic Usage

To benchmark an operator, run the following command:

```
$ python run.py --op gemm
```

## Install as a library

To install as a library:

```
$ pip install -e .
# in your own benchmark script
import tritonbench
from tritonbench.utils import parser
op_args = parser.parse_args()
addmm_bench = tritonbench.load_opbench_by_name("addmm")(op_args)
addmm_bench.run()
```

## Submodules

We depend on the following projects as a source of customized Triton or CUTLASS kernels:

* (CUDA, HIP) [generative-recommenders](https://github.com/facebookresearch/generative-recommenders)
* (CUDA, HIP) [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
* (CUDA, HIP) [tilelang](https://github.com/tile-ai/tilelang)
* (CUDA) [xformers](https://github.com/facebookresearch/xformers)
* (CUDA) [flash-attention](https://github.com/Dao-AILab/flash-attention)
* (CUDA) [FBGEMM](https://github.com/pytorch/FBGEMM)
* (CUDA) [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
* (HIP) [AITer](https://github.com/ROCm/aiter)


## License
TritonBench is BSD 3-Clause licensed, as found in the LICENSE file.
