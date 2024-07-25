import torch
import torch.nn as nn
import torch.jit as jit
import numpy as np


class SnakeActivation(jit.ScriptModule):
    def __init__(self, initial_a=0.2, learnable=True):
        super().__init__()
        if learnable:
            self.a = nn.Parameter(torch.tensor(initial_a, dtype=torch.float32))
        else:
            self.register_buffer('a', torch.tensor(initial_a, dtype=torch.float32))

    @jit.script_method
    def forward(self, x):
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2


class FlexibleSnakeActivation(jit.ScriptModule):
    """
    this flexible version allows 
    - multiple values of `a` for different channels/num_features for the learnable option.
    - sample a from a uniformdistribution (a_base, a_max) for the learnable option.
    """
    def __init__(self, num_features:int, dim:int, a_base=0.2, learnable=True, a_max=0.5):
        super().__init__()
        assert dim in [1, 2], '`dim` supports 1D and 2D inputs.'

        if learnable:
            if dim == 1:  # input dim: (b d l); like time series
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1))  # (1 d 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            elif dim == 2:  # input dim: (b d h w); like 2d images
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1, 1))  # (1 d 1 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        else:
            self.register_buffer('a', torch.tensor(a_base, dtype=torch.float32))

    @jit.script_method
    def forward(self, x):
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2