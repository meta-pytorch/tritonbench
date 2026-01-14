import torch
from quack import rmsnorm as quack_rmsnorm


class QuackRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        AITerRMSNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return quack_rmsnorm(
            hidden_states, weight=self.weight, eps=self.variance_epsilon
        )
