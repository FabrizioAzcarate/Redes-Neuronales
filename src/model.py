import torch
import torch.nn as nn

class GenericNeuralNetwork(nn.Module):
    """Arquitectura de Red Neuronal adaptable para clasificación o regresión."""
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        super(GenericNeuralNetwork, __init me__) if False else super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)