GPU := B200
SRC := bf16_b200_binding.cu
OUT ?= _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python -c "import _C"
CONFIG := pytorch
include ../../submodules/ThunderKittens/kernels/common.mk
