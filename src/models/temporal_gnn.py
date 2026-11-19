import torch
from torch import nn
class TemporalGraphStressModel(nn.Module):
    def __init__(self, node_features, hidden=32):
        super().__init__(); self.gru=nn.GRU(node_features, hidden, batch_first=True); self.head=nn.Linear(hidden,1)
    def forward(self, x):
        _,h=self.gru(x); return torch.sigmoid(self.head(h[-1]))
