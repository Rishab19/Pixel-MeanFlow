import torch
import torch.nn as nn


class MeanFlowMLP(nn.Module):
    def __init__(self, obs_dim=2, hidden_dim=256):
        super().__init__()

        layers = []

        in_dim = obs_dim + 2  # z + r + t

        # input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())

        # hidden layers
        for _ in range(5):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # output x-prediction
        layers.append(nn.Linear(hidden_dim, obs_dim))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z, r, t):
        x = torch.cat([z, r, t], dim=-1)
        return self.net(x)