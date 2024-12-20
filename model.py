import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    
    def __init__(self, input_dim=1, output_dim=1) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_dim, 16)
        self.l2 = nn.Linear(16, 8)
        self.l3 = nn.Linear(8, output_dim)
    
    def forward(self,x):
        x = F.sigmoid(self.l1(x))
        x = F.sigmoid(self.l2(x))
        x = F.sigmoid(self.l3(x))
        return x
    
    

