# TritonBench Inputs

In TritonBench, users can customize the input data to run. Here is an overview of the CLI options related to inputs.

| Option                | Usage                                                                                                                                                                                                                |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--input-id`          | Input ID to run, starting from 0.      Default is 0.                                                                                                                                                                 |
| `--num-inputs`        | Number of inputs to run. By default, run all available inputs.                                                                                                                                                       |
| `--input-sample-mode` | Input sampling mode. 'first-k' (default) uses the first k inputs starting from `--input-id`.  "'equally-spaced-k' selects k equally spaced inputs from the entire input range, where k is specified by --num-inputs. |
| `--input-loader`      | Specify a json file to load inputs from the input json file.                                                                                                                                                         |


