# Numerics Checking with TritonBench

TritonBench supports numerics checking and export for accuracy checking. Every operator backend can export its output with a given input.
In the forward mode, it will compare the forward pass output tensors.
In the backward mode, it will compare the gradients of input tensors that requires gradients.

## Compare numerics of different compiler backends with `--metrics accuracy`

`--metrics accuracy` requires the operator declares a backend as the baseline and will compare other backend numerics against it.
Users can use `--baseline <BACKEND_NAME>` to specify the baseline backend. If unspecified, TritonBench will use the backend decorated by `@register_benchmark(baseline=True)`.

By default, TritonBench uses `torch.testing.assert_close()` API [link](https://docs.pytorch.org/docs/stable/testing.html),
which will set different `rtol` and `atol` thresholds. For example, for `bfloat16` dtype, `rtol` is `1.6e2` and `atol` is `1e-5`.
`--metrics accuracy` will return `1` when the numeric matches the baseline, and `0` when it does not.
We provide CLI options `--rtol` and `--atol` for users to tune these thresholds, they are both `None` by default, which will use the default values used by PyTorch.

If users want to create their own numeric checking methods, they can override the accuracy checking metric like in this [code example](https://github.com/meta-pytorch/tritonbench/blob/9a4bbc7070b134fb274114018ac02b38fcfd4ba7/tritonbench/operators/vector_exp/operator.py#L88).

We force all backends of one operator to comply to the same numeric checking criteria.

## Compare numerics on different hardware platforms with export output

When comparing numerics on different devices, we provide `--export [input | output | both]` and `--export-dir <DIR>` options.
Users need to first export the tensor outputs to one directory on one device, then run the same command to export the output on the second device,
and finally copy the two directories under the same filesystem for comparison.

We provide a simple script to compare two directories:

```
python benchmarks/numeric_check/run.py --a <DIR_ON_DEVICE_A> --b <DIR_ON_DEVICE_B>
```

For cross-device numeric checking, we only support the default threshold using `torch.testing.all_close()`.
