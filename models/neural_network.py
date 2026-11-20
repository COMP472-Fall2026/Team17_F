import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.0,
             hidden_size=256, use_batch_norm=False):
        super().__init__()
        # The first and last layer sizes depend on the dataset.
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, features):
        # Give each possible label a score.
        return self.layers(features)
