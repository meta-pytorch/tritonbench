# TritonBench Input Data

In TritonBench, users can customize the input data to run. Here is an overview of the CLI options related to inputs.

| Option                | Usage                                                                                                                                                                                                                |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--input-id`          | Input ID to run, starting from 0.      Default is 0.                                                                                                                                                                 |
| `--num-inputs`        | Number of inputs to run. By default, run all available inputs.                                                                                                                                                       |
| `--input-sample-mode` | Input sampling mode. 'first-k' (default) uses the first k inputs starting from `--input-id`.  "'equally-spaced-k' selects k equally spaced inputs from the entire input range, where k is specified by --num-inputs. |
| `--input-loader`      | Specify a json file (or wildcard pattern) to load inputs from input json file(s).                                                                                                                                    |


## Input Data Collection

We keep a set of input data in the [data/input_configs](https://github.com/meta-pytorch/tritonbench/tree/main/tritonbench/data/input_configs) directory.
The input data is organized by model names and is in json format. User can specify the input config by `--input-loader <path-to-input-json>`.
TritonBench will generate synthetic inputs based on the input config.

### Wildcard Matching

The `--input-loader` flag supports wildcard patterns to load and merge multiple input config files at once. This is useful when you want to benchmark across multiple input configurations.

**Supported wildcards:** `*`, `?`, `[`, `]`

#### Examples

**Single file (standard usage):**
```bash
--input-loader path/to/input_configs/fb/ads_omnifm_v4/domain_experts_gemm.json
```

**Wildcard matching multiple files:**
```bash
# Load all gemm input configs from a directory
--input-loader 'path/to/input_configs/fb/ads_omnifm_v4/*_gemm.json'
```

#### Behavior

When multiple files match a wildcard pattern:

1. **Preload**: All matching JSON files are loaded
2. **Validation**: All files must have the same `tritonbench_ops` in their metadata. If files have different ops (e.g., mixing `gemm` and `addmm` configs), the job will terminate with an error
3. **Merge & Dedupe**: Inputs from all files are merged, with duplicate inputs (based on the `inputs` field) removed

#### Example Output

```
[input-loader] Merged 15 input config files with 221 total unique inputs
```

#### Error Case

If you try to merge files with different `tritonbench_ops`, you will see an error like:

```
RuntimeError: All input config files must have the same tritonbench_ops, but found different ops:
  - path/to/file1_addmm.json: ('addmm',)
  - path/to/file2_gemm.json: ('gemm',)
```

To fix this, make your wildcard pattern more specific (e.g., use `*_gemm.json` instead of `*.json`).
