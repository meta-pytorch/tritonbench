"""Diode (ML model for pruning autotuning configs) utils for TritonBench operators."""

import diode.torch_diode.config as diode_config
from diode.torch_diode.choices import DiodeInductorChoices
from diode.torch_diode.models.triton_gemm.model import (
    GEMMModelV2,
    MODEL_CONFIGS,
)
from diode.torch_diode.registry import register, get_registry
import logging
from torch._inductor.virtualized import V
from torch._inductor.choices import InductorChoices

DIODE_MODEL_CONFIGS_VERSION = "v3_12_04_2025"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_diode_model(topk: int = 1, expand_search_space: bool = True) -> tuple[int, bool]:
    logger.info("[DIODE][TritonBench] Setup Diode model.")

    old_topk = diode_config.topk
    old_expand_search_space = diode_config.expand_search_space

    diode_config.topk = topk
    diode_config.expand_search_space = expand_search_space

    gemm_diode_model: GEMMModelV2 = GEMMModelV2(
        model_config=MODEL_CONFIGS[DIODE_MODEL_CONFIGS_VERSION]
    )
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
