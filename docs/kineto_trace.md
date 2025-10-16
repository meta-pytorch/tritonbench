# Kineto Trace Analysis with TritonBench



## Example 1: Kineto Trace Analysis

TritonBench supports generating a Kineto trace file for each `<input, impl>` pair.
We use the following command to generate 6 Kineto traces, as it is running 2 inputs(`--num-inputs 2`) with 3 impls (`flash_v3,cudnn,triton_tutorial_flash_v2`).

```
$ python run.py --op flash_attention --num-inputs 2 --metrics kineto_trace --only flash_v3,cudnn,triton_tutorial_flash_v2

  (Batch, Heads, SeqLen, Dhead)                                      flash_v3-kineto_trace                                cudnn_90100-kineto_trace                                      triton_tutorial_flash_v2-kineto_trace
-------------------------------  ---------------------------------------------------------  ------------------------------------------------------  -------------------------------------------------------------------------
               (4, 48, 128, 64)  /tmp/tritonbench/flash_attention/kineto_traces/flash_v3_0  /tmp/tritonbench/flash_attention/kineto_traces/cudnn_0  /tmp/tritonbench/flash_attention/kineto_traces/triton_tutorial_flash_v2_0
               (4, 48, 256, 64)  /tmp/tritonbench/flash_attention/kineto_traces/flash_v3_1  /tmp/tritonbench/flash_attention/kineto_traces/cudnn_1  /tmp/tritonbench/flash_attention/kineto_traces/triton_tutorial_flash_v2_1
```

The output table shows the directory where the Kineto trace file is stored.

Opening the trace file with Chrome Trace Viewer, we need to first separate the profiling iteration with the warm-up iterations.
The profiling iteration runs after all warm-up iteraions and is labeled by `ProfilerStep#<number>`.

![Kineto Trace](https://ossci-datasets.s3.us-east-1.amazonaws.com/tritonbench/docs/_static/img/kineto_trace_fig_1.png "Kineto Trace - Global View")

Zooming into the profile iteration, we find two GPU kernels launched. The first one corresponds to the L2 Cache flush to clear the cache.
The second one corresponds to the actual computation kernel, which is from CUDNN in this flash_attention operator.

![Kineto Trace](https://ossci-datasets.s3.us-east-1.amazonaws.com/tritonbench/docs/_static/img/kineto_trace_fig_2.png "Kineto Trace - Zoomed into Profile Iteration")

## Example 2: Kineto Trace with CUDA Graph enabled

If the operator supports CUDA Graph and CUPTI, we can generate Kineto trace with CUDA Graph enabled. To do that, simply combine `--cudagraph` with `--metrics kineto_trace`.
Here is an example command:

```
$ python run.py --op flash_attention --num-inputs 1 --metrics kineto_trace --only triton_tutorial_flash_v2 --cudagraph

  (Batch, Heads, SeqLen, SeqLen_KV, Dhead)                                        triton_tutorial_flash_v2-kineto_trace
------------------------------------------  ---------------------------------------------------------------------------
                     (4, 48, 128, 128, 64)  /tmp/tritonbench_xzhao9/bf16_flash_attention_fwd/triton_tutorial_flash_v2_0
                                   average

```



![Kineto Trace](https://ossci-datasets.s3.us-east-1.amazonaws.com/tritonbench/docs/_static/img/kineto_trace_cudagraph_fig_1.png "Kineto Trace - CUDA Graph launch")
