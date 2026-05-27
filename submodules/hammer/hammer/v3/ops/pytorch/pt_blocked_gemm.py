# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

from __future__ import annotations

from typing import List, Optional

import torch


def pytorch_blocked_gemm(
    A_list: List[List[torch.Tensor]],
    W_list: List[List[torch.Tensor]],
    bias_list: Optional[List[torch.Tensor]] = None,
) -> List[List[torch.Tensor]]:
    """Blocked GEMM with K blocking using PyTorch ops.

    Computes C[i][j] = sum_k A[i][k] @ W[j][k]^T + bias[j]

    A_list[i][k] has shape (M_i, K_k).
    W_list[j][k] has shape (N_j, K_k).
    Output C_list[i][j] has shape (M_i, N_j).

    Internally concatenates across K, M, N dimensions, performs a single
    matmul (or addmm if bias), and splits the output. PyTorch autograd
    handles the backward pass natively.
    """
    num_a = len(A_list)
    num_b = len(W_list)
    num_k = len(A_list[0])

    # Transpose W to get B: B[j][k] has shape (K_k, N_j)
    B_list = [
        [W_list[j][k].t().contiguous() for k in range(num_k)] for j in range(num_b)
    ]

    # Concat across K dimension
    A_rows = [torch.cat(A_list[i], dim=1) for i in range(num_a)]
    B_cols = [torch.cat(B_list[j], dim=0) for j in range(num_b)]

    # Concat across M and N dimensions
    A_concat = torch.cat(A_rows, dim=0)
    B_concat = torch.cat(B_cols, dim=1)

    # Single GEMM
    if bias_list is not None:
        bias_concat = torch.cat(bias_list)
        C_concat = torch.addmm(bias_concat, A_concat, B_concat)
    else:
        C_concat = torch.mm(A_concat, B_concat)

    # Split output back into blocks
    M_sizes = [A_list[i][0].shape[0] for i in range(num_a)]
    N_sizes = [W_list[j][0].shape[0] for j in range(num_b)]
    C_rows = torch.split(C_concat, M_sizes, dim=0)
    return [list(torch.split(row, N_sizes, dim=1)) for row in C_rows]
