"""Diode (ML model for pruning autotuning configs) utils for TritonBench operators."""

import json
import logging

from torch._inductor.choices import InductorChoices
from torch._inductor.virtualized import V
from tritonbench.utils.env_utils import is_fbcode

if is_fbcode():  # Diode not available in OSS
    import diode.torch_diode.config as diode_config
    from diode.torch_diode.choices import DiodeInductorChoices
    from diode.torch_diode.models.triton_gemm.encode_features import FeatureVersion
    from diode.torch_diode.models.triton_gemm.model import (
        GEMMModelV2,
        MODEL_CONFIGS,
        ModelConfig,
    )
    from diode.torch_diode.registry import get_registry, register

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def deserialize_model_config(json_str: str) -> "ModelConfig":
    """Deserialize a JSON string into a ModelConfig instance.

    Args:
        json_str: JSON string representing a ModelConfig. The JSON should have
            the following fields:
            - model_name (str): Manifold model path
            - n_hidden_layers (int): Number of hidden layers
            - dropout_rate (float): Dropout rate
            - template_op_pairs (list[list[str, str]]): Template/op pairs
              (JSON arrays, converted to tuples)
            - supported_devices (list[str] | null): Supported GPU devices
            - feature_version (str): Feature version string (e.g. "v4")
            - is_production (bool, optional): Production bucket flag. Default: True
            - description (str, optional): Human-readable description. Default: ""

    Returns:
        A ModelConfig instance.

    Raises:
        json.JSONDecodeError: If the input is not valid JSON.
        KeyError: If a required field is missing.
        ValueError: If feature_version is not a valid FeatureVersion enum value.
    """
    data = json.loads(json_str)
    return ModelConfig(
        model_name=data["model_name"],
        n_hidden_layers=data["n_hidden_layers"],
        dropout_rate=data["dropout_rate"],
        template_op_pairs=[tuple(pair) for pair in data["template_op_pairs"]],
        supported_devices=data.get("supported_devices"),
        feature_version=FeatureVersion(data["feature_version"]),
        is_production=data.get("is_production", True),
        description=data.get("description", ""),
    )


def setup_diode_model(
    diode_version: str,
    topk: int = 1,
    expand_search_space: bool = True,
    model_config: "ModelConfig | None" = None,
) -> tuple[int, bool]:
    logger.info("[DIODE][TritonBench] Setup Diode model.")

    old_topk = diode_config.topk
    old_expand_search_space = diode_config.expand_search_space

    diode_config.topk = topk
    diode_config.expand_search_space = expand_search_space

    if model_config is None:
        model_config = MODEL_CONFIGS[diode_version]

    gemm_diode_model: GEMMModelV2 = GEMMModelV2(model_config=model_config)
    register(gemm_diode_model)

    V.set_choices_handler(DiodeInductorChoices())

    return old_topk, old_expand_search_space


def teardown_diode_model(old_configs):
    logger.info("[DIODE][TritonBench] Teardown Diode model.")

    old_topk, old_expand_search_space = old_configs
    diode_config.topk = old_topk
    diode_config.expand_search_space = old_expand_search_space
    get_registry().clear()
    V.set_choices_handler(InductorChoices())
