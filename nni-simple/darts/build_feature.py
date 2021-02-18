import torch
import torch.nn as nn

class LinearNormal(nn.Module):
    def __init__(self, out_filters):
        super().__init__()
        self.out_filters = out_filters
        self.net = nn.Sequential(
            nn.Linear(self.out_filters, self.out_filters)
        )
    def forward(self, x):
        return self.net(x)

class LinearRelu(nn.Module):
    def __init__(self, out_filters):
        super().__init__()
        self.out_filters = out_filters
        self.net = nn.Sequential(
            nn.Linear(self.out_filters, self.out_filters),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class LinearTanh(nn.Module):
    def __init__(self, out_filters):
        super().__init__()
        self.out_filters = out_filters
        self.net = nn.Sequential(
            nn.Linear(self.out_filters, self.out_filters),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)
