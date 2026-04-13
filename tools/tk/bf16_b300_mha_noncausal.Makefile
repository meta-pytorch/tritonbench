GPU := B300
SRC := ../../submodules/ThunderKittens/kernels/attention/bf16_b300_mha_noncausal/bf16_b300_mha_noncausal.cu
OUT ?= _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python -c "import _C"
CONFIG := pytorch
include ../../submodules/ThunderKittens/kernels/common.mk
