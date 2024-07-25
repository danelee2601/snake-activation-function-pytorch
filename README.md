# Snake Activation Function (PyTorch)

The Snake Activation Function [1] is a novel activation function for neural networks, designed to introduce non-linearity and enhance the model's representational power. This README provides an overview of the PyTorch implementation of the Snake Activation Function.

## Features

- **Learnable and Unlearnable Options**: The Snake Activation Function implementation offers both learnable and unlearnable options for $a$. This allows flexibility in choosing whether the parameters of the activation function should be updated during training or remain fixed.
- **Initialization for Snake**: [1] suggests $0.2 \leq a \leq a_{max}$ for standard tasks such as image classification. However, for tasks with expected periodicity, larger $a$, usually from $5$ to $50$ tend to work well.
- **Learnable $a$ is a default setting**: Learnable $a$ works better than constant $a$ in general . 


## Usage

To use the Snake Activation Function in your PyTorch project, 
import the `SnakeActivation` class from `snake.py`:

```python
from snake import SnakeActivation
snake = SnakeActivation()
```

## Example

Here's an example of how to use the Snake Activation Function in a PyTorch model:

```python
import torch
import torch.nn as nn
from snake import SnakeActivation

class Net1D(nn.Module):
    def __init__(self):
        super(Net1D, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1)
        self.snake = FlexibleSnakeActivation(16, dim=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: (b c l)
        """
        x = self.snake(self.conv1(x))
        x = self.conv2(x)
        return x

model = Net1D()

x = torch.rand(4, 3, 100)
out = model(x)  # (4 16 100)
```
For more details on the Snake Activation Function and its implementation, please refer to the `snake.py` file.


## Variant: Flexible Snake
The original snake function has a single $a$. The function can be expanded by having $a$ per channel to allow different non-linearities for different channels, which can potentially lead to better performance. 
For example, consider a 1D input tensor with dimensions $(\text{batch size}, \text{num channels}, \text{length}) = (b, c, l)$. The function can have $c$ instances of $a$. For 2D input tensor with dimensions $(\text{batch size}, \text{num channels}, \text{height}, \text{width}) = (b, c, h, w)$, it's the same, i.e., $c$ instances of $a$.

Examples:
```python
from snake import FlexibleSnakeActivation

# Example for 1D input with dimensions (batch_size, n_channels, length)
class Net1D(nn.Module):
    def __init__(self):
        super(Net1D, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1)
        self.snake = FlexibleSnakeActivation(16, dim=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: (b c l)
        """
        x = self.snake(self.conv1(x))
        x = self.conv2(x)
        return x

# Example for 2D input with dimensions (batch_size, n_channels, height, width)
class Net2D(nn.Module):
    def __init__(self):
        super(Net2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.snake = FlexibleSnakeActivation(16, dim=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: (b c h w)
        """
        x = self.snake(self.conv1(x))
        x = self.conv2(x)
        return x


model_1d = Net1D()
model_2d = Net2D()

x_1d = torch.rand(4, 3, 100)
out = model_1d(x)  # (4 16 100)

x_2d = torch.rand(4, 3, 32, 32)
out = model_2d(x)  # (4 16 32 32)
```

I found that `FlexibleSnakeActivation` generally results in better performance than `SnakeActivation`.

## Reference
[1] Ziyin, Liu, Tilman Hartwig, and Masahito Ueda. "Neural networks fail to learn periodic functions and how to fix it." Advances in Neural Information Processing Systems 33 (2020): 1583-1594.