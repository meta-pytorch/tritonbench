# TritonBench Input Data

In TritonBench, users can customize the input data to run. Here is an overview of the CLI options related to inputs.

| Option                | Usage                                                                                                                                                                                                                |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--input-id`          | Input ID to run, starting from 0.      Default is 0.                                                                                                                                                                 |
| `--num-inputs`        | Number of inputs to run. By default, run all available inputs.                                                                                                                                                       |
| `--input-sample-mode` | Input sampling mode. 'first-k' (default) uses the first k inputs starting from `--input-id`.  "'equally-spaced-k' selects k equally spaced inputs from the entire input range, where k is specified by --num-inputs. |
| `--input-loader`      | Specify a json file to load inputs from the input json file.                                                                                                                                                         |


## Input Data Collection

We keep a set of input data in the [data/input_configs](https://github.com/meta-pytorch/tritonbench/tree/main/tritonbench/data/input_configs) directory.
The input data is organized by model names and is in json format. User can specify the input config by `--input-loader <path-to-input-json>`.
TritonBench will generate synthetic inputs based on the input config.