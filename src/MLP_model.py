import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────
# ARQUITECTURA MLP (Multilayer Perceptron)
# ──────────────────────────────────────────────────────────────────────
class SolarMLP(nn.Module):
    """
    Red Neuronal Densa (MLP) de Alta Complejidad.
    Incluye múltiples capas ocultas y Dropout.
    """
    def __init__(self, input_size, hidden_size=256, dropout=0.3):
        super(SolarMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.relu3 = nn.ReLU()
        
        self.output = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        # Aplanar (batch, seq, feat) -> (batch, seq * feat)
        x = x.view(x.size(0), -1)
        
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        
        return self.output(x)
