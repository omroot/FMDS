import torch
import torch.nn as nn


class SteinScoreNetwork(nn.Module):
    def __init__(self, input_dimension: int = 32,
                 hidden_dimension: int = 64,
                 number_hidden_layers: int = 3,
                 output_dimension: int = 32):
        super(SteinScoreNetwork, self).__init__()

        # Define the input layer
        self.input_layer = nn.Linear(input_dimension, hidden_dimension)

        # Dynamically create hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dimension, hidden_dimension) for _ in range(number_hidden_layers)]
        )

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, x):
        # Pass through input layer
        x = torch.relu(self.input_layer(x))

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        # Pass through output layer
        x = self.output_layer(x)
        return x