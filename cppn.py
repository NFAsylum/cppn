import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.linear import Linear
from torch.nn.modules.container import ModuleList

class CPPN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, hidden_layers=4) -> None:
        super().__init__()

        self.first: Linear = nn.Linear(input_dim, hidden_dim)
        self.hidden: ModuleList = nn.ModuleList(modules=[
            nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)
        ])
        self.output: Linear = nn.Linear(hidden_dim, 3)

    def forward(self, coords) -> Tensor:
        x: Tensor = self.first(coords)
        x = torch.sin(x)
        for layer in self.hidden:
            x = layer(x)
            x = torch.tanh(x)
        x = self.output(x)

        return torch.sigmoid(x)