from functools import partial
from typing import List, Union

import torch
from torch._inductor.lowering import make_pointwise, register_lowering

# Some internal torch.compile details
from torch._inductor.virtualized import ops
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    _score_mod_signature,
    and_masks,
    or_masks,
)

Tensor = torch.Tensor


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def generate_alibi_bias(H: int) -> _score_mod_signature:
    """Returns an alibi bias score_mod given the number of heads H

    Args:
        H: number of heads

    Returns:
        alibi_bias: alibi bias score_mod
    """

    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (kv_idx - q_idx) * scale
        return score + bias

    return alibi_mod


def length_to_offsets(lengths: List[int], device: Union[str, torch.device]) -> Tensor:
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature, offsets: Tensor
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        q_logical = q_idx - offsets[document_id[q_idx]]
        kv_logical = kv_idx - offsets[document_id[kv_idx]]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask

    return doc_mask_mod


def generate_prefix_lm_mask(prefix_length: int) -> _mask_mod_signature:
    """Generates a prefix LM causal attention mask.

    Args:
        prefix_length: The length of the prefix.

    Note:
        This mask allows full attention within the prefix (first PREFIX_LENGTH tokens)
        and causal attention for the rest of the sequence.
    """

    def prefix_mask(b, h, q_idx, kv_idx):
        return kv_idx < prefix_length

    prefix_lm_causal_mask = or_masks(prefix_mask, causal_mask)
    prefix_lm_causal_mask.__name__ = f"prefix_lm_causal_mask_{prefix_length}"
    return prefix_lm_causal_mask


def generate_sliding_window(window_size: int) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size

    sliding_window_mask = and_masks(sliding_window, causal_mask)
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask


@torch.library.custom_op("approx::tanh", mutates_args=())
def _tanh_approx(inp: Tensor) -> Tensor:
    return torch.tanh(inp)


@_tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


def _tanh_approx_lowering(inp):
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)


register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)


class _TanhApprox(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ops.approx.tanh(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result = output
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * (1 - result * result)

    @staticmethod
    def vmap(info, in_dims, x):
        return torch.tanh(x), 0


_tanh_approx = _TanhApprox.apply


def generate_tanh_softcap(soft_cap: int, approx: bool = False) -> _score_mod_signature:
    """Returns an tanh bias score_mod given the number of heads H

    Args:
        soft_cap: The soft cap value to use for normalizing logits
        approx: Whether to use the `tanh.approx.` ptx instruction

    Returns:
        tanh_softcap: score_mod
    """
    tanh = _tanh_approx if approx else torch.tanh

    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return soft_cap * tanh(score / soft_cap)

    prefix = "tanh_softcap_approx" if approx else "tanh_softcap"
    tanh_softcap.__name__ = f"{prefix}_{soft_cap}"

    return tanh_softcap
