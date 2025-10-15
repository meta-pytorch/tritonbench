import numpy as np
import torch

import triton


if torch.version.hip is not None:
    configs = [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 16,
                "BLOCK_K": 256,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 16,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "BLOCK_K": 256,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
    ]
else:
    configs = [
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ]


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_full_amd_config_space(use_splitk: bool):
    configs = []

    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [8]
    waves_per_eu_range = [0, 1, 2, 4]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for waves_per_eu in waves_per_eu_range:
                            base_config_dict = {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "GROUP_M": group_m,
                                "waves_per_eu": waves_per_eu,
                                "kpack": 2,
                            }
                            config_dicts = []
                            if use_splitk:
                                max_k_pow2 = np.int64(np.log2(block_k))
                                split_k_range = [2**i for i in range(max_k_pow2)]
                                for split_k in split_k_range:
                                    config_dicts.append(
                                        {
                                            **base_config_dict,
                                            "SPLIT_K": split_k,
                                        }
                                    )
                            else:
                                config_dicts.append(base_config_dict)
                            for config_dict in config_dicts:
                                configs.append(
                                    triton.Config(
                                        config_dict,
                                        num_warps=num_warps,
                                        num_stages=2,
                                        pre_hook=(
                                            init_to_zero("C") if use_splitk else None
                                        ),
                                    )
                                )
    return configs


def get_tileir_configs():
    block_mnk_range = [16, 32, 64, 128, 256]
    occ_range = [1, 2]
    num_ctas_range = [1, 2]
    configs = []
    for block_m in block_mnk_range:
        for block_n in block_mnk_range:
            for block_k in block_mnk_range:
                for occ in occ_range:
                    for num_ctas in num_ctas_range:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "GROUP_M": 8,
                                    "occupancy": occ,
                                },
                                num_ctas=num_ctas,
                            )
                        )
    return configs
