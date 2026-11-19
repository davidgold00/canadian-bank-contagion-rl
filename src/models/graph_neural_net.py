import torch
from torch import nn
class LightweightGraphNet(nn.Module):
    def __init__(self, in_features, hidden=32, out_features=1):
        super().__init__(); self.lin1=nn.Linear(in_features,hidden); self.lin2=nn.Linear(hidden,out_features)
    def forward(self, X, A):
        H=torch.relu(A @ self.lin1(X)); return torch.sigmoid(self.lin2(H))
