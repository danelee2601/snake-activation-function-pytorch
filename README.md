# Snake Activation Function (PyTorch)

The Snake Activation Function [1] is a novel activation function for neural networks, designed to introduce non-linearity and enhance the model's representational power. This README provides an overview of the PyTorch implementation of the Snake Activation Function.

## Features

- **Learnable and Unlearnable Options**: The Snake Activation Function implementation offers both learnable and unlearnable options for $a$. This allows flexibility in choosing whether the parameters of the activation function should be updated during training or remain fixed.
- **Initialization for Snake**: [1] suggests $0.2 \leq a \leq a_{max}$ for standard tasks such as image classification. However, for tasks with expected periodicity, larger $a$, usually from $5$ to $50$ tend to work well.


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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
        self.snake = SnakeActivation()

    def forward(self, x):
        x = self.snake(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the Net model
model = Net()

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train your model
# ...

```

For more details on the Snake Activation Function and its implementation, please refer to the `snake.py` file.

## Reference
[1] Ziyin, Liu, Tilman Hartwig, and Masahito Ueda. "Neural networks fail to learn periodic functions and how to fix it." Advances in Neural Information Processing Systems 33 (2020): 1583-1594.