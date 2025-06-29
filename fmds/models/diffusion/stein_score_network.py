import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler


class SteinScoreNetwork(nn.Module):
    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: int = 64,
                 number_hidden_layers: int = 3,
                 output_dimension: int = 32,
                 dropout_prob: float = 0.25):
        """
        Score network that takes concatenated input [x | sigma].

        Args:
            input_dimension: Dimensionality of input data + 1 (x + sigma).
            hidden_dimension: Size of hidden layers.
            number_hidden_layers: Number of hidden layers.
            output_dimension: Output dimension (should match data dimensionality).
            dropout_prob: Dropout probability.
        """
        super(SteinScoreNetwork, self).__init__()

        self.input_layer = nn.Linear(input_dimension, hidden_dimension)
        self.dropout = nn.Dropout(dropout_prob)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dimension, hidden_dimension),
                nn.SELU(),
                nn.Dropout(dropout_prob)
            ) for _ in range(number_hidden_layers)
        ])

        self.output_layer = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.selu(self.input_layer(x))
        x = self.dropout(x)

        for layer in self.hidden_layers:
            x = layer(x)

        return self.output_layer(x)

