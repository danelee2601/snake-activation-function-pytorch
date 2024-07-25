import torch
import torch.nn as nn

class SnakeActivation(nn.Module):
    def __init__(self, initial_a=0.2, learnable=True):
        super(SnakeActivation, self).__init__()
        if learnable:
            self.a = nn.Parameter(torch.tensor(initial_a, dtype=torch.float32))
        else:
            self.a = torch.tensor(initial_a, dtype=torch.float32)

    def forward(self, x):
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2
