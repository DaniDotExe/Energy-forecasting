import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────
# ARQUITECTURA MLP (Multilayer Perceptron)
# ──────────────────────────────────────────────────────────────────────
class SolarMLP(nn.Module):
    """
    Red Neuronal Densa (MLP) para predicción mensual.
    Recibe una ventana de meses y devuelve el kWh del siguiente mes.
    """
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super(SolarMLP, self).__init__()
        
        # El input_size será: (INPUT_WINDOW * N_FEATURES)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, 1) # Salida: 1 solo valor (kWh del mes siguiente)
        )
        
    def forward(self, x):
        # x tiene forma (batch, seq_len, features)
        # Para un MLP plano, debemos "aplanar" la secuencia
        x = x.view(x.size(0), -1) # (batch, seq_len * features)
        return self.network(x)
