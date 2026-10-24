#!/usr/bin/python3

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.c_fc(x), approximate="tanh")
        return self.drop(self.c_proj(x))
