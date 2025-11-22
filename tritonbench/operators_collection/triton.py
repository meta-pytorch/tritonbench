"""
Operator Collection Triton
================================

Return list of operators and their triton backends.
"""

from tritonbench.utils.metadata_utils import get_metadata


def get_triton_operators():
    triton_operators = get_metadata("cuda_kernels")
    result = {}
    for op in triton_operators:
        for backend in triton_operators[op]:
            if (
                triton_operators[op][backend]
                and "tags" in triton_operators[op][backend]
                and "triton" in triton_operators[op][backend]["tags"]
            ):
                if not op in result:
                    result[op] = []
                result[op].append(backend)
    return result
