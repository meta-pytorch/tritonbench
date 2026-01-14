"""
Benchmark mojo_matmul with modular nightly.
To install modular nightly:
pip install --pre modular --index-url https://dl.modular.com/public/nightly/python/simple/
"""

import argparse
import json
import logging
import os
import sys
from os.path import abspath, exists
from typing import Dict, List


def setup_tritonbench_cwd():
    original_dir = abspath(os.getcwd())

    for tritonbench_dir in (
        ".",
        "../../../tritonbench",
    ):
        if exists(tritonbench_dir):
            break

    if exists(tritonbench_dir):
        tritonbench_dir = abspath(tritonbench_dir)
        os.chdir(tritonbench_dir)
        sys.path.append(tritonbench_dir)
    return original_dir


setup_tritonbench_cwd()

from typing import Callable

import max.graph as mg
import torch
from max import driver, engine
from max.graph import DeviceRef, Graph, ops, TensorType, TensorValue
from max.graph.type import DType, Shape, ShapeLike
from tritonbench.operators import load_opbench_by_name
from tritonbench.utils.parser import get_parser
from tritonbench.utils.triton_op import register_benchmark


def promote_mojo_tensor_to_fp32(mojo_tensor, dtype):
    input_type = TensorType(
        dtype=dtype, shape=mojo_tensor.shape, device=DeviceRef.GPU()
    )
    with mg.Graph("mojo_to_fp32", input_types=(input_type,)) as g:
        (inp,) = g.inputs
        out = ops.cast(inp, dtype=DType.float32)
        g.output(out)
    session = engine.InferenceSession(devices=[driver.Accelerator()])
    model = session.load(g)
    output = model.execute(mojo_tensor)
    return output


def demote_numpy_to_mojo_tensor_dtype(numpy_array, dtype):
    with mg.Graph("mojo_to_dtype") as g:
        inp = ops.constant(numpy_array, dtype=DType.float32, device=DeviceRef.GPU())
        out = ops.cast(inp, dtype=dtype)
        g.output(out)
    session = engine.InferenceSession(devices=[driver.Accelerator()])
    model = session.load(g)
    output = model.execute()
    return output[0]


MOJO_DTYPE_MAPPING = {
    "bf16": DType.bfloat16,
    "fp32": DType.float32,
    "fp16": DType.float16,
}
MOJO_DEVICE_MAPPING = {
    "cuda": DeviceRef.GPU,
    "cpu": DeviceRef.CPU,
}
MOJO_DRIVER_DEVICE_MAPPING = {
    "cuda": driver.Accelerator,
    "cpu": driver.CPU,
}


def mojo_matmul(operator, a, b, bias) -> Callable:
    precision = operator.precision
    device = operator.device
    mojo_dtype = MOJO_DTYPE_MAPPING[precision]
    mojo_device = MOJO_DEVICE_MAPPING[device]
    mojo_driver_device = MOJO_DRIVER_DEVICE_MAPPING[device]
    a_numpy = a.cpu().float().numpy()
    b_numpy = b.T.cpu().float().numpy()
    a_mojo_cuda = driver.Tensor.from_numpy(a_numpy).to(mojo_driver_device())
    b_mojo_cuda = driver.Tensor.from_numpy(b_numpy).to(mojo_driver_device())
    a_mojo_bf16 = demote_numpy_to_mojo_tensor_dtype(a_numpy, mojo_dtype)
    b_mojo_bf16 = demote_numpy_to_mojo_tensor_dtype(b_numpy, mojo_dtype)
    input_types = (
        TensorType(dtype=mojo_dtype, shape=a_numpy.shape, device=mojo_device()),
        TensorType(dtype=mojo_dtype, shape=b_numpy.shape, device=mojo_device()),
    )
    with mg.Graph("mojo_matmul", input_types=input_types) as g:
        a_val, b_val = g.inputs
        c_val = ops.matmul(a_val, b_val.T)
        g.output(c_val)
    session = engine.InferenceSession(devices=[driver.Accelerator()])
    model = session.load(g)
    outputs = model.execute(a_mojo_bf16, b_mojo_bf16)
    output_func = lambda: model.execute(a_mojo_bf16, b_mojo_bf16)
    return output_func


if __name__ == "__main__":
    args = [
        "--op",
        "gemm",
        "--only",
        "aten_matmul,mojo_matmul",
        "--precision",
        "bf16",
        "--m",
        "512",
        "--n",
        "8192",
        "--k",
        "5376",
    ] + sys.argv[1:]
    gemm_opbench_cls = load_opbench_by_name("gemm")
    parser = get_parser(args)
    tb_args, extra_args = parser.parse_known_args(args)
    gemm_opbench = gemm_opbench_cls(tb_args, extra_args)
    gemm_opbench.add_benchmark(bm_func_name="mojo_matmul", bm_callable=mojo_matmul)
    gemm_opbench.run()
    metrics = gemm_opbench.output
    print(metrics)
    # TODO: promote the output to fp32 for numerics check
    # y_torch = torch.from_numpy(promote_mojo_tensor_to_fp32(outputs[0], dtype=DType.bfloat16)[0].to_numpy())
