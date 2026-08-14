---
description: Run the CompileIQ ptxas autotuner with a short search
argument-hint: "[tritonbench-config.yaml] [extra run.py flags]"
allowed-tools: Bash, Read, Glob
---

Run the CompileIQ ptxas autotuner (`benchmarks/compileiq_autotune`) in short-search mode.

Arguments (all optional): `$ARGUMENTS`
- First token, if it ends in `.yaml`, is the tritonbench run config **file name** (not a path) — it must exist in `benchmarks/run_config/`. Defaults to `gemm_config_3.yaml` (`DEFAULT_CONFIG_FILE` in `benchmarks/compileiq_autotune/shim.py`).
- Any remaining tokens are passed through to `run.py` verbatim (e.g. `--search-space /path/to/space.bin`, `--results-csv /tmp/foo.csv`).

## Steps

1. Resolve the config. If a `.yaml` was given, verify `benchmarks/run_config/<name>` exists; if it doesn't, list the close matches in that directory and stop. If no config was given, use the default.

2. Sanity-check the environment before burning a search on a broken setup:
   - `nvidia-smi -L` — the search needs at least one visible GPU.
   - `python -c "import compileiq"` — the tuner imports `compileiq.ciq`, which is not vendored in this repo.

   If either fails, report exactly what's missing and stop rather than running the search.

3. Run the objective-function smoke test first (one baseline tritonbench run, no ptxas controls). This catches a bad config or a broken kernel in ~a minute instead of mid-search:

   ```bash
   cd /data/users/xzhao9/tritonbench && \
   python -m benchmarks.compileiq_autotune.run --test \
     --tritonbench-config <config>
   ```

   If it exits non-zero, show the error and stop.

4. Run the short search in the background (it still takes many minutes — `pool_size=8` evaluations at up to `TRITONBENCH_COMPILEIQ_TIMEOUT`, default 1800s, each), teeing output to a log:

   ```bash
   cd /data/users/xzhao9/tritonbench && \
   python -m benchmarks.compileiq_autotune.run --short \
     --tritonbench-config <config> \
     <extra flags> 2>&1 | tee /tmp/tritonbench_compileiq_search/short-<config-name>-<timestamp>.log
   ```

   Use `run_in_background: true` and poll, rather than blocking on a single long call. Create `/tmp/tritonbench_compileiq_search` first if it doesn't exist.

5. When it finishes, report:
   - The best result line (`[tritonbench_compileiq] Best result:`).
   - The results CSV path (defaults to `/tmp/tritonbench_compileiq_search/result-compileiq.csv`) and the top few rows by the objective metric.
   - The metric being optimized and its direction — `tflops` is maximized, `latency` is minimized; it's read from `--metrics` in the run config.

   If the run failed, quote the relevant traceback or the `Error running Tritonbench` block from the log; do not summarize it away.

## Notes

- Short mode overrides `--generations` to 1 with `pool_size=8`, `cull_size=4` (see `search()` in `benchmarks/compileiq_autotune/run.py`). Passing `--generations` alongside `--short` has no effect — say so if the user asks for both.
- The default search space is a `manifold://` URI for the ptxas 13.3 knobs, downloaded once into `/tmp/tritonbench_compileiq_search/`. The first run needs working `manifold` CLI access; later runs reuse the local copy.
- Manifold result upload and best-config extraction only happen under MAST (`MAST_HPC_JOB_*` env vars set). A local short run leaves results in the CSV only.
