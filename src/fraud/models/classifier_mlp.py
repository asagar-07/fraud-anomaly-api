from __future__ import annotations

import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    """
    Simple MLP for binary fraud classification.

    Input:  (batch, 30) scaled numeric features
    Output: (batch, 1) logits (NOT sigmoid)
    """

    def __init__(self, input_dim: int = 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # First hidden layer, 30 input features to 64 neurons
            nn.BatchNorm1d(64), # Batch normalization for better training
            nn.ReLU(), # Activation function
            nn.Dropout(p=0.30), # Dropout for regularization, disables 30% of neurons, not active during inference
            nn.Linear(64, 32), # Second hidden layer, 64 input features to 32 neurons   
            nn.BatchNorm1d(32), # Batch normalization for better training
            nn.ReLU(), # Activation function
            nn.Dropout(p=0.20), # Dropout for regularization, disables 20% of neurons, not active during inference
            nn.Linear(32, 1), # Output layer (logits for binary classification)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits