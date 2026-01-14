import torch
from aiter.ops.triton.rmsnorm import rms_norm as aiter_rms_norm


class AITerRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        AITerRMSNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return aiter_rms_norm(hidden_states, self.weight, self.variance_epsilon)
