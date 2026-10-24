#!/usr/bin/python3

import attention
import mlp
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = attention.CausalSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = mlp.MLP(n_embd, bias, dropout)

    def forward(self, x):
        return x + self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))
