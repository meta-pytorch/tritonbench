# TritonBench Bisection

This doc describes the regression detection and bisection service in TritonBench.

## Overview

The bisect workflow uses git bisect to automatically find the first commit that introduced a performance or feature regression. It:

1. Establishes a baseline by running benchmarks on the known "good" commit
2. Uses git bisect to binary search through commits between good and bad
3. At each step, builds Triton, runs benchmarks, and compares against baseline
4. Reports the first commit that introduced the regression

## Usage

### Via GitHub Actions

Trigger the workflow manually from the Actions tab:

1. Go to Actions → "Bisect Benchmark Regression"
2. Click "Run workflow"
3. Fill in the required parameters:
   - `good_commit`: A known good Triton commit (no regression)
   - `bad_commit`: A known bad Triton commit (has regression)
   - `triton_repo`: Triton repository (default: triton-lang/triton)
   - `benchmark_name`: Benchmark to run (default: nightly)
   - `operator`: Specific operator to test (optional, empty for all)
   - `metric`: Metric to compare (latency, tflops, speedup)
   - `regression_threshold`: Threshold for regression detection (0.1 = 10%)
   - `runner_type`: GPU runner (h100 or mi350)

### Local Usage

```bash
export GOOD_COMMIT="abc123..."
export BAD_COMMIT="def456..."
export BENCHMARK_NAME="nightly"
export METRIC="latency"
export REGRESSION_THRESHOLD="0.1"
export SETUP_SCRIPT="/path/to/setup.sh"
export WORKSPACE_DIR="/workspace"

bash .ci/bisect/run-bisect.sh
```

## Scripts

### run-bisect.sh

Main bisect orchestration script that:
- Clones the Triton repository
- Runs baseline benchmarks on the good commit
- Executes git bisect with automatic good/bad detection
- Outputs the first bad commit with details

### compare_results.py

Python script for comparing benchmark results:

```bash
python .ci/bisect/compare_results.py \
    --baseline baseline_results.json \
    --current current_results.json \
    --metric latency \
    --threshold 0.1
```

Exit codes:
- 0: No regression (current is within threshold)
- 1: Regression detected

## Output

Results are saved to `bisect-output/`:
- `bisect.log`: Full log of the bisect process
- `baseline_results.json`: Benchmark results from good commit
- `step_N/results.json`: Results from each bisect step
- `bisect_result.json`: Final result with first bad commit

## Example

To find which Triton commit caused a 15% latency regression in the gemm operator:

```yaml
good_commit: "a1b2c3d4..."  # Last known good commit
bad_commit: "e5f6g7h8..."   # First known bad commit
operator: "gemm"
metric: "latency"
regression_threshold: "0.15"
```

## Notes

- The bisect process can take several hours depending on the commit range
- Build failures are treated as "bad" commits
- Benchmark failures are also treated as "bad" commits
- The workflow has a 12-hour timeout to accommodate long bisect sessions