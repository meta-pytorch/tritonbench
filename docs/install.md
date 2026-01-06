# TritonBench Installation Guide

We manage TritonBench install with `install.py`.

## Basic Usage

If you run `python install.py` without any argument, it will do the following:


1. If PyTorch is missing, install the latest PyTorch nightly package.
2. Install other dependencies from requirements.txt and requirements_numpy.txt.
3. Checkout all submodules

## Advanced Usage

`install.py` supports arguments that enable users to install optional third-party kernel libraries.

- `--liger`: Install [liger-kernel-nightly](https://github.com/linkedin/Liger-Kernel) package (Triton).
- `--fa3`: Install Flash Attention 3 (CUTLASS).
- `--fa2`: Install Flash Attention 2 (CUTLASS).
- `--fbgemm`: Install FBGEMM GPU kernels.
- `--mslk`: Install MSLK kernels (CUTLASS and Triton).
- `--jax`: Install JAX (Pallas and Mosaic).
- `--tk`: Install [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).
- `--tile`: Install [Tile Lang](https://github.com/tile-ai/tilelang).
- `--xformers`: Install [XFormers](https://github.com/facebookresearch/xformers).
- `--aiter`: Install [AITer](https://github.com/ROCm/aiter).
- `--all`: Install all of the above.
