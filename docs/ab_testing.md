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
- `--side-b`: Parameter string for configuration B

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

A/B test output consists of three sections:

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
