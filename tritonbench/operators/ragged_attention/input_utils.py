import torch


def generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate the sparse-length distribution shared by HSTU benchmarks."""
    if sparsity == 0.0:
        return torch.zeros(size=(size,), device=device, dtype=torch.int)
    if sparsity == 1.0:
        return torch.full(
            size=(size,),
            fill_value=max_seq_len,
            device=device,
            dtype=torch.int,
        )
    if sparsity >= 0.5:
        min_seq_len = int((2 * sparsity - 1.0) * max_seq_len)
        return torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size,),
            device=device,
            dtype=torch.int,
            generator=generator,
        )
    sparse_max_seq_len = int(2 * sparsity * max_seq_len)
    return torch.randint(
        low=0,
        high=sparse_max_seq_len,
        size=(size,),
        device=device,
        dtype=torch.int,
        generator=generator,
    )
