import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.utils._pytree import tree_map


def input_cast(cond, action, example_inputs):
    """Traverse the input batch pytree, and cast tensor with `action` if it satisfies `cond`."""
    if isinstance(example_inputs, torch.Tensor) and cond(example_inputs):
        return action(example_inputs)
    elif isinstance(example_inputs, (tuple, list, dict)):
        return tree_map(lambda x: input_cast(cond, action, x), example_inputs)
    elif (
        example_inputs is None
        or isinstance(example_inputs, str)
        or isinstance(example_inputs, int)
        or isinstance(example_inputs, float)
    ):
        # Do not touch primitive types
        return example_inputs
    elif isinstance(example_inputs, torch.Tensor):
        return example_inputs
    elif isinstance(example_inputs, torch.nn.Module):
        return example_inputs
    elif isinstance(example_inputs, BlockMask):
        return example_inputs
    # Triton TensorDescriptor is a non-primitive runtime object that should be
    # passed through unchanged during input casting.
    elif (
        example_inputs.__class__.__name__ == "TensorDescriptor"
        and hasattr(example_inputs, "base")
        and hasattr(example_inputs, "block_shape")
    ):
        return example_inputs
    # FlexAttention passes around functions as inputs
    elif callable(example_inputs):
        return example_inputs
    else:
        raise RuntimeError(f"Unsupported input type: {type(example_inputs)}")


def input_filter(cond, example_inputs):
    """Traverse the input batch pytree, and return the first element that satisfies cond."""
    if isinstance(example_inputs, dict):
        return input_filter(cond, example_inputs.items())
    elif isinstance(example_inputs, (tuple, list)):
        return next(
            item for item in example_inputs if input_filter(cond, item) is not None
        )
    elif cond(example_inputs):
        return example_inputs
    else:
        return None
