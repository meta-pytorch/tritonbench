"""TileIR utils for TritonBench."""

import itertools

import triton


def generate_exhaustive_tileir_configs():
    return [
        triton.Config(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K,
                "occupancy": occ,
            },
            num_ctas=num_ctas,
        )
        for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
            [16, 32, 64, 128, 256], repeat=3
        )
        for occ in [1, 2]
        for num_ctas in [1, 2]
    ]


def convert_triton_to_tileir_configs(triton_configs) -> list[triton.Config]:
    tileir_configs = set()
    for config in triton_configs:
        if "occupancy" not in config.kwargs:
            for occ in [1, 2]:
                tileir_configs.add(
                    triton.Config(
                        {**config.kwargs, "occupancy": occ},
                        num_warps=config.num_warps,
                        num_stages=config.num_stages,
                        num_ctas=config.num_ctas,
                        maxnreg=config.maxnreg,
                        pre_hook=config.pre_hook,
                        ir_override=config.ir_override,
                    )
                )

    for config in tileir_configs:
        for num_ctas in [1, 2]:
            tileir_configs.add(
                triton.Config(
                    config.kwargs,
                    num_warps=config.num_warps,
                    num_stages=config.num_stages,
                    num_ctas=num_ctas,
                    maxnreg=config.maxnreg,
                    pre_hook=config.pre_hook,
                    ir_override=config.ir_override,
                )
            )

    return list(tileir_configs)


def prune_duplicate_configs(configs, named_args, **kwargs) -> list[triton.Config]:
    """
    Prune duplicate configs, i.e. those with all the same parameters except for num_warps and num_stages.
    """
    pruned_configs = set()
    for config in configs:
        config.num_warps = None
        config.num_stages = None
        pruned_configs.add(config)
    return list(pruned_configs)
