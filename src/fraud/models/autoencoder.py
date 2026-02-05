'''
    •   Accept input tensor shape: (batch, 30)
	•	Output reconstruction tensor shape: (batch, 30)
	•	No sigmoid on output (these are scaled real values)
	•	Keep architecture symmetric: 30 → 16 → 8 → 16 → 30
	•	Use ReLU in hidden layers, linear output layer
''' 
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

class FraudAutoencoder(nn.Module):
    """
    Autoencoder for fraud detection.

    Input:  (batch, 30) scaled numeric features
    Output: (batch, 30) reconstructed features
    """

    def __init__(self, input_dim: int = 30, hidden_dims: Tuple[int, int] = (16, 8)) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), # First hidden layer, 30 input features to 16 neurons
            nn.ReLU(), # Activation function
            nn.Linear(hidden_dims[0], hidden_dims[1]), # Second hidden layer, 16 input features to 8 neurons
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]), # First hidden layer of decoder, 8 input features to 16 neurons
            nn.ReLU(), # Activation function
            nn.Linear(hidden_dims[0], input_dim), # Output layer, reconstruct back to 30 features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  # reconstructed features