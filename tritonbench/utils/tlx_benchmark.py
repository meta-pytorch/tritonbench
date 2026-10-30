import importlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, is_dataclass, replace
from functools import partial
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

logger: logging.Logger = logging.getLogger(__name__)


def check_accuracy(op, fn, baseline_fn):
    from torch.utils._pytree import tree_flatten

    try:
        actual = fn()
        if op.mode.value == "bwd":
            actual = op._clone_gradients(actual, mode="BWD")
        reference = baseline_fn()
        if op.name == "tlx_flash_attention_varlen":
            reference = op._varlen_unpad(reference)
        if op.mode.value == "bwd":
            reference = op._clone_gradients(reference, mode="BWD")
        got, got_spec = tree_flatten(actual)
        ref, ref_spec = tree_flatten(reference)
        assert got_spec == ref_spec
        assert any(isinstance(t, torch.Tensor) for t in got)
        for output, expected in zip(got, ref):
            if output is None or expected is None:
                assert output is None and expected is None
                continue
            assert torch.isfinite(output).all() and torch.isfinite(expected).all()
            rel = (
                0.05
                if op.name == "addmm"
                else (0.008 if expected.dtype == torch.bfloat16 else 0.001)
            )
            if op.tb_args.bitwise:
                assert torch.equal(output, expected)
            else:
                requested_rtol, requested_atol = op._tlx_requested_tolerances
                rtol = rel if requested_rtol is None else requested_rtol
                atol = (
                    rel * expected.abs().max().item()
                    if requested_atol is None
                    else requested_atol
                )
                torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
        return True
    except AssertionError as error:
        logger.warning("TLX L1 accuracy check failed: %s", error)
        return False


def _record_compilation(op, **event):
    op._tlx_evidence.append(
        {
            "kind": "compile",
            "input_id": op._cur_input_id,
            "kernel": str(event["src"]),
            "cache_hit": event["cache_hit"],
            "times": event["times"],
            "metadata": event["metadata"],
            "artifacts": event["metadata_group"],
        }
    )


def _record_autotune(op, **event):
    op._tlx_evidence.append(
        {
            "kind": "autotune",
            "input_id": op._cur_input_id,
            "kernel": str(event["fn"]),
            "key": str(event["key"]),
            "best_config": str(event["best_config"]),
            "benchmarked_configs": len(event["configs_timings"]),
            "configs_timings": {str(c): t for c, t in event["configs_timings"].items()},
            "duration_s": event["duration"],
            "cache_hit": event["cache_hit"],
        }
    )


def _monitor_template_choices(op, config):
    from triton.language.extra.tlx.inductor.choices import TLXInductorChoices

    finalize = TLXInductorChoices._finalize_template_configs

    def checked_choices(self, *args, **kwargs):
        choices = finalize(self, *args, **kwargs)
        if config.triton.tlx_mode == "force":
            op._tlx_evidence.append(
                {
                    "kind": "template_choices",
                    "input_id": op._cur_input_id,
                    "count": len(choices),
                    "choices": [
                        {"template": c.template.uid, "config": c.params.to_kwargs()}
                        for c in choices
                    ],
                }
            )
            if len(choices) != 1:
                raise NotImplementedError(
                    f"Heuristic gap: forced torchTLX emitted {len(choices)} configs"
                )
        return choices

    TLXInductorChoices._finalize_template_configs = checked_choices


def _monitor_mm_selection(op):
    module = importlib.import_module("triton.tlx.ops.kernels.mm.sm90")
    select_path = module._select_path
    seen = set()

    def recorded_selection(space, key, launchers):
        selected = select_path(space, key, launchers)
        if key not in seen:
            seen.add(key)
            op._tlx_evidence.append(
                {
                    "kind": "selected_kernel",
                    "input_id": op._cur_input_id,
                    "space": space,
                    "key": str(key),
                    "selected": selected,
                    "candidate_paths": list(launchers),
                }
            )
        return selected

    module._select_path = recorded_selection


def configure_flex_bwd(op, config_name):
    module = importlib.import_module(
        "triton.language.extra.tlx.inductor.flex_attention_templates"
    )
    selected = tuple(map(int, config_name.split(",")))
    if selected not in module._SM90_BWD_CONFIGS:
        raise ValueError(f"Unknown existing H100 flex backward config: {selected}")
    template = module.sm90_flex_attention_backward_template
    append_choice = template.maybe_append_choice

    def append_selected(*args, **kwargs):
        if (kwargs["BLOCK_N"], kwargs["NUM_KV_BUFFERS"]) != selected:
            return
        choices = kwargs["choices"]
        before = len(choices)
        result = append_choice(*args, **kwargs)
        added = choices[before:]
        op._tlx_evidence.append(
            {
                "kind": "flex_template_choice",
                "input_id": op._cur_input_id,
                "template": template.name,
                "tlx_mode": inductor_config.triton.tlx_mode,
                "max_autotune": inductor_config.max_autotune,
                "max_autotune_gemm": inductor_config.max_autotune_gemm,
                "count": len(added),
                "choices": [str(choice) for choice in added],
                "config": {
                    key: kwargs[key]
                    for key in (
                        "BLOCK_N",
                        "NUM_KV_BUFFERS",
                        "BLOCK_Q",
                        "SHARED_KV",
                        "TLX_KV_SUM",
                        "num_warps",
                        "num_stages",
                    )
                },
            }
        )
        if len(added) != 1:
            raise NotImplementedError(
                "Selected H100 flex backward config is unavailable"
            )
        return result

    template.maybe_append_choice = append_selected


def prepare_run(op):
    op._tlx_requested_tolerances = op.tb_args.rtol, op.tb_args.atol
    for metric in ("accuracy", "latency", "speedup", "setup_time", "first_call_time"):
        if metric not in op.required_metrics:
            op.required_metrics.append(metric)
    cache = tempfile.mkdtemp(prefix="tritonbench-tlx-")
    os.environ["TORCHINDUCTOR_TLX_MODE"] = "off"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache, "inductor")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "0"
    os.environ["MPLBACKEND"] = "Agg"
    op._tlx_artifacts = Path(cache)
    if not op.tb_args.output_dir:
        op.tb_args.output_dir = str(op._tlx_artifacts / "results")
    op._tlx_evidence = []
    import triton

    triton.knobs.compilation.listener = partial(_record_compilation, op)
    triton.knobs.autotuning.listener = partial(_record_autotune, op)
    import torch._inductor.config as config

    config.triton.tlx_mode = None
    config.max_autotune = False
    config.max_autotune_gemm = False
    if "tlx_ops_mm" in op._only:
        _monitor_mm_selection(op)
    if any(name.startswith("torch_tlx_") for name in op._only):
        importlib.import_module("triton.language.extra.tlx.inductor.registry")
        torch._logging.set_logs(output_code=True)
        _monitor_template_choices(op, config)
    logger.warning(
        "TLX L2 cache policy: %s",
        json.dumps(
            {
                "cache_root": cache,
                "max_autotune": False,
                "default_tlx_mode": None,
                "cold_timing": "synchronized provider setup plus first call, before warmup and graph capture",
            }
        ),
    )


def _contract_label(i, contract):
    if "m" in contract:
        return f"{i}: ({contract['m']}, {contract['n']}, {contract['k']}) {contract['dtype']} A{contract['a_strides']} B{contract['b_strides']}"
    if "kv_max_seqlen" in contract:
        return f"{i}: B{contract['batch']} Q{contract['min_seqlen']}..{contract['max_seqlen']} KV{contract['kv_min_seqlen']}..{contract['kv_max_seqlen']} H{contract['heads']} D{contract['head_dim']} {contract['dtype']} causal={contract['causal']}"
    return f"{i}: {tuple(contract.values())}"


def plot_results(op):
    from tritonbench.utils.plot_utils import plot_benchmark_results

    output_dir = Path(op.tb_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "x_val": str(x),
            "input_contract": op._tlx_contracts[i],
            "providers": {p: asdict(m) for p, m in values.items()},
        }
        for i, (x, values) in enumerate(op.output.result)
    ]
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    (output_dir / "evidence.json").write_text(
        json.dumps(
            op._tlx_evidence,
            indent=2,
            default=lambda v: asdict(v) if is_dataclass(v) else str(v),
        )
    )

    output = replace(
        op.output,
        result=[
            (_contract_label(i, op._tlx_contracts[i]), values)
            for i, (_, values) in enumerate(op.output.result)
        ],
    )
    plot_kwargs = {}
    if op.name == "flex_attention" and op.torch_tlx_config is not None:
        plot_kwargs["plot_title"] = (
            f"H100 flex backward ({op.torch_tlx_config}): "
            "stock compiled flex vs forced torchTLX"
        )
    plot_benchmark_results(output, op.tb_args, op.name, show_plot=False, **plot_kwargs)
    for metric in ("setup_time", "first_call_time"):
        plot_benchmark_results(
            output,
            op.tb_args,
            op.name,
            y_metric=metric,
            y_label=f"{metric} (ms)",
            show_plot=False,
            **plot_kwargs,
        )
