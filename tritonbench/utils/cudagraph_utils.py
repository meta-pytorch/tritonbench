import threading
from functools import partial
from typing import Callable, Optional

import torch
from torch.profiler import profile, ProfilerActivity


class CudaGraphConfig:
    stream: Optional[torch.cuda.Stream] = None
    graph: Optional[torch.cuda.CUDAGraph] = None
    num_inputs_per_iter: int = -1
    num_kernels_per_input: int = -1

    @staticmethod
    def generate(use_cuda_graphs: bool):
        if use_cuda_graphs:
            return CudaGraphConfig()
        return None

    def __init__(self):
        if CudaGraphConfig.stream is None:
            CudaGraphConfig.stream = torch.cuda.Stream()
        if CudaGraphConfig.graph is None:
            CudaGraphConfig.graph = torch.cuda.CUDAGraph()

    def get_stream(self):
        """
        Get the stream for capturing the CUDA graph.
        """
        return CudaGraphConfig.stream

    def get_graph(self):
        """
        Get the CUDA graph for capturing the CUDA graph.
        """
        return CudaGraphConfig.graph

    def reset_graph(self):
        """
        Reset the CUDA graph for capturing the CUDA graph.
        """
        CudaGraphConfig.graph.reset()

    def get_num_kernels(self, fn: Callable):
        """
        Estimate the number of GPU kernels executed in a function using PyTorch
        profiler.
        """

        def trace_handler(prof, num_kernels_ptr, trace_event):
            event_list = prof.events()
            count = 0
            for e in event_list:
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    count += 1

            if count == 0:
                # If we do not see any CUDA events, we estimate the number of
                # kernels from the number of cudaLaunchKernel events.
                for e in event_list:
                    if "cudaLaunchKernel" in e.name:
                        count += 1

            num_kernels_ptr[0] = count
            # Signal the main thread that the trace handler is done
            trace_event.set()

        stream = self.get_stream()
        num_kernels_ptr = [0]
        trace_event = threading.Event()

        with profile(
            activities=[ProfilerActivity.CUDA],
            on_trace_ready=partial(
                trace_handler,
                num_kernels_ptr=num_kernels_ptr,
                trace_event=trace_event,
            ),
        ):
            with torch.cuda.stream(stream):
                fn()

        torch.cuda.synchronize()
        # Wait for the trace handler to finish
        trace_event.wait()

        num_kernels = num_kernels_ptr[0]
        if num_kernels >= self.num_inputs_per_iter:
            # Keep track of the number of kernels per input
            num_kernels_per_input = num_kernels // self.num_inputs_per_iter
            if CudaGraphConfig.num_kernels_per_input == -1:
                CudaGraphConfig.num_kernels_per_input = num_kernels_per_input
            else:
                CudaGraphConfig.num_kernels_per_input += num_kernels_per_input
                CudaGraphConfig.num_kernels_per_input //= 2
                CudaGraphConfig.num_kernels_per_input = max(
                    CudaGraphConfig.num_kernels_per_input, 1
                )

        # If the number of kernels is too low compared to the previous
        # estimations (this can cause by a profiler failure), we estimate the
        # number of kernels from the number of inputs and average number of
        # kernels per input
        return max(
            num_kernels,
            self.num_inputs_per_iter * CudaGraphConfig.num_kernels_per_input,
        )


class CudaGraphError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)
