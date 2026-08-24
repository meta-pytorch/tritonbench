# TritonBench A/B Testing

## Overview

The A/B testing feature allows you to compare two different configurations in a single run, helping you quickly evaluate the performance impact of different parameter settings.

## Basic Usage

### Command Format
```bash
python run.py --op <operator> --side-a="<configuration A>" --side-b="<configuration B>"
```

### Parameters
- `--op`: Name of the operator to test (single operator only)
- `--side-a`: Parameter string for configuration A
- `--side-b`: Parameter string for configuration B (optional)

### Single-Side Runs
`--side-b` is optional. With only `--side-a`, that configuration runs by itself and
its latency samples are analyzed, which is a quick way to see how noisy a
configuration is before comparing anything against it:
```bash
python run.py --op vector_add --side-a="--warmup 25"
```
`--side-b` on its own is an error: it always needs a `--side-a` to compare against.

## Configuration Types

### 1. Global Parameter Testing
Global parameters are tritonbench-level settings that affect the entire benchmark behavior:

```bash
# Test different warmup parameters
python run.py --op vector_add --side-a="--warmup 25" --side-b="--warmup 100"

# Test different precision settings
python run.py --op flash_attention --side-a="--precision fp16" --side-b="--precision fp32"

# Test different device settings
python run.py --op gemm --side-a="--device cuda" --side-b="--device cpu"
```

### 2. Operator-Specific Parameter Testing
Each operator has its own specific parameters:

```bash
# Test different head counts for flex_attention
python run.py --op flex_attention --side-a="--n-heads-q 8" --side-b="--n-heads-q 16"

# Test different matrix sizes for gemm
python run.py --op gemm --side-a="--m 1024 --n 1024 --k 1024" --side-b="--m 2048 --n 2048 --k 2048"
```

### 3. Mixed Parameter Testing
You can test both global and operator-specific parameters simultaneously:

```bash
# Test both warmup and data type
python run.py --op flash_attention --side-a="--warmup 50 --dtype fp16" --side-b="--warmup 100 --dtype bf16"

# Global precision + operator-specific parameters
python run.py --op vector_add --side-a="--precision fp16 --n 1000000" --side-b="--precision fp32 --n 5000000"
```

## Parameter Formats

### Equal Sign Format After --side Flag
You must use the equal sign after the --side-a or --side-b flag:
```bash
python run.py --op flex_attention --side-a="--warmup 25" --side-b="--warmup 100"
```

### Default Configuration
If you provide an empty string `""`, it represents the default configuration:
```bash
# Compare custom configuration against default
python run.py --op vector_add --side-a="--warmup 100 --precision fp16" --side-b=""

# Compare default against custom configuration
python run.py --op flash_attention --side-a="" --side-b="--dtype bf16 --batch-size 16"
```

### Multiple Parameters
```bash
python run.py --op flash_attention --side-a="--warmup 50 --dtype fp16 --batch-size 8" --side-b="--warmup 100 --dtype bf16 --batch-size 16"
```

## Output Format

A/B test output consists of four sections (a single-side run prints only the
latency analysis):

### 1. Configuration Analysis
Shows differences between the two configurations:
```
Configuration Differences:
  warmup         : 25              → 100
  precision      : fp16            → fp32
```

### 2. Performance Summary
Shows average performance changes for each backend and metric:
```
Performance Summary
----------------------------------------------------------------------

torch_add:
  latency     : +37.8% avg [-22.2% to +96.4%]
  gbps        : -27.4% avg [-49.1% to +28.6%]

triton_add:
  latency     : +41.5% avg [-12.5% to +96.9%]
  gbps        : -29.3% avg [-49.2% to +14.3%]
```

### 3. Detailed Comparison
Shows specific numerical comparisons for each metric across different input sizes and backends:
```
Metric: latency
Backend        x_val               Config A    Config B    Difference
-----------------------------------------------------------------------
torch_add      4096                0.009       0.007       -22.2%
               8192                0.007       0.007        +0.0%
               16384               0.008       0.007       -12.5%
...
```

### 4. Latency Analysis
Printed whenever the `latency` metric was collected, from the raw per-iteration
samples that `do_bench` keeps (see `tritonbench/components/do_bench/latency_analysis.py`).
A/B runs disable the 1.5x IQR outlier filter that `do_bench` normally applies, so
these statistics describe the full distribution -- including the tails, which is
what makes the dispersion and hypothesis tests meaningful. The `min`/`max` in the
results table above therefore span the untrimmed range and can look far wider
than in a non-A/B run of the same benchmark.
Each side gets descriptive statistics (min/max/mean/median/stddev/stderr, CV, IQR)
plus a Student-t and a bootstrap confidence interval. When both sides are present,
Shapiro-Wilk decides which test to run: Welch's t-test with Cohen's d if both
samples look normal, otherwise the Mann-Whitney U test with a rank-biserial
correlation. Either way the percent change is reported with a bootstrap CI:
```
triton_add @ x_val=4096:
  Config A (n=200):
    min=0.0071  max=0.0093  mean=0.0080  median=0.0080
    stddev=0.0003  stderr=0.0000  CV=3.75%  IQR=0.0004 [Q1=0.0078, Q3=0.0082]
    95% CI (t): [0.0080, 0.0081]  bootstrap mean: [0.0080, 0.0081]  bootstrap median: [0.0079, 0.0081]
  Config B (n=200):
    ...
    Shapiro-Wilk: Config A: W=0.9946, p=0.6840 (normal); Config B: W=0.9953, p=0.7956 (normal)
    Welch's t-test: t=23.413, dof=396.2, p=<1e-4 (significant at alpha=0.05)
    Cohen's d: 2.341 (large)
    Percent change (Config B vs Config A): +4.78% [95% CI: +4.36%, +5.17%]
```

## JSON Output

`--output-json <path>` writes the whole A/B run to a single file instead of the
per-run file a normal benchmark produces (both sides share one `--output-json`,
so writing it per side would just have B overwrite A):

```bash
python run.py --op vector_add --side-a="--warmup 25" --side-b="--warmup 100" --output-json ab.json
```

```json
{
    "side-a": { },
    "side-b": { },
    "ab-comparison": { }
}
```

`side-b` and `ab-comparison` are only present when `--side-b` is specified.

Each side holds its own configuration and results:

| Key | Contents |
| --- | --- |
| `config` | The `--side-x` args as given |
| `global_args` | Effective tritonbench globals: the command line's, overridden by the side's |
| `op_args` | Operator-specific args of the side |
| `op_name`, `op_mode` | Operator and mode that ran |
| `metrics` | One entry per `(backend, x_val)` cell, keyed `tritonbench_<op>[<backend>-<x_val>]` |

A `metrics` entry holds the metrics collected for that cell -- each reported as
a single p50 -- followed by the descriptive statistics of the raw latency
samples behind them (omitted when the `latency` metric was not collected, or
when there were too few samples to analyze):

```json
{
    "config": ["--warmup", "25"],
    "metrics": {
        "tritonbench_vector_add[triton_add-4096]": {
            "latency": 0.006272,
            "gbps": 7.836,
            "n": 2140,
            "min": 0.005952,
            "max": 0.015904,
            "mean": 0.006552,
            "median": 0.006272,
            "stddev": 0.000499,
            "stderr": 0.0000108,
            "cv": 7.629,
            "q1": 0.006176,
            "q3": 0.007040,
            "iqr": 0.000864,
            "confidence": 0.95,
            "mean_ci": [0.006531, 0.006573],
            "bootstrap_mean_ci": [0.006532, 0.006574],
            "bootstrap_median_ci": [0.006240, 0.006304]
        }
    }
}
```

`ab-comparison` holds the same four report sections that are printed to the log:

| Key | Contents |
| --- | --- |
| `config_differences` | `{param: {"side-a": ..., "side-b": ...}}` |
| `x_vals`, `backends`, `metrics` | The compared scope (the intersection of both sides) |
| `performance_summary` | Per backend and metric: `avg`/`min`/`max_improvement` and `count` |
| `detailed_comparison` | One row per `(metric, backend, x_val)` with both values and `pct_change` |
| `latency_comparison` | Per `(backend, x_val)` normality test, hypothesis test, effect size and percent change CI; omitted when the `latency` metric was not collected |

Statistics that are undefined for the samples at hand (e.g. a percent change
against a zero mean) are written as `null` rather than the JSON-invalid `NaN`.

With `--mode fwd,bwd` each mode gets its own file (`ab_fwd.json`, `ab_bwd.json`).

## Error Handling

The system automatically handles the following error conditions:
- Configuration parsing failures: Provides clear error messages
- Benchmark execution failures: Shows specific error reasons
- Empty results: Detects and reports empty result issues
- Parameter parsing errors: Issues warnings and uses default values

## Limitations

1. **Single Operator Restriction**: A/B testing only supports single operators, not multi-operator comparisons
2. **Common Inputs**: Both configurations must have overlapping input sizes for comparison
3. **Common Backends**: Only backends that exist in both configurations will be compared
4. **Sequential Execution**: Still investigating how and how much running A/B sequentially will affect B's performance

## Troubleshooting

### Configuration Parsing Failures
Ensure parameter string format is correct, especially proper use of quotes:
```bash
# Correct
python run.py --op vector_add --side-a="--warmup 25" --side-b="--warmup 100"

# Wrong: missing quotes
python run.py --op vector_add --side-a=--warmup 25 --side-b=--warmup 100
```

### No Common Input Sizes or Backends
Check that both configurations can run successfully and produce comparable results.
