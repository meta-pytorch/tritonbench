from itertools import product


def llama_shapes():
    # batch sizes * seq lengths
    BS = [2**i for i in range(0, 17)]
    # attn: wqkv, wo; ffn: w13, w2
    KN = [
        (4096, 12288),
        (4096, 4096),
        (4096, 22016),
        (11008, 4096),
        (8192, 1280),
        (1024, 8192),
        (8192, 7168),
        (3584, 8192),
        (16384, 2304),
        (2048, 16384),
        (16384, 13312),
        (6656, 16384),
    ]
    return [(bs, n, k, None) for bs, (k, n) in product(BS, KN)]
