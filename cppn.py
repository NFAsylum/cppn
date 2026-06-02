import torch
from torch import Tensor
from torch.nn import Module, Linear, ModuleList

class CPPN(Module):
    def __init__(self, input_dim=2, hidden_dim=32, hidden_layers=4) -> None:
        super().__init__()

        self.first: Linear = Linear(input_dim, hidden_dim)
        self.hidden: ModuleList = ModuleList([
            Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)
        ])
        self.output: Linear = Linear(hidden_dim, 3)

    def forward(self, coords) -> Tensor:
        x: Tensor = self.first(coords)
        x = torch.sin(x)
        for layer in self.hidden:
            x = layer(x)
            x = torch.tanh(x)
        x = self.output(x)

        return torch.sigmoid(x)