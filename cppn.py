import torch
from torch import Tensor
from torch.nn import Module, Linear, ModuleList
from torch.nn.init import normal_

class CPPN(Module):
    # hidden_dim=128 with hidden_layers=8 will likely cause model collapse
    def __init__(self, input_dim=7, hidden_dim=64, hidden_layers=6, weight_sigma=1.0, output_channels=3) -> None:
        super().__init__()

        self.first: Linear = Linear(input_dim, hidden_dim)
        self.hidden: ModuleList = ModuleList([
            Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)
        ])
        self.output: Linear = Linear(hidden_dim, output_channels)

        self._init_weights(weight_sigma)

    def _init_weights(self, sigma: float) -> None:
        for layer in [self.first, *self.hidden, self.output]:
            normal_(layer.weight, mean=0.0, std=sigma)

    def forward(self, coords) -> Tensor:
        x: Tensor = self.first(coords)
        x = torch.sin(x)
        activations = [torch.sin, torch.tanh, lambda x: torch.exp(-x**2), torch.tanh]
        for i, layer in enumerate(self.hidden):
            x = layer(x)

            x = activations[i % len(activations)](x)

        x = self.output(x)

        return torch.sigmoid(x)