import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            fwd_expansion,
            layers,
            max_len,
            pad_idx,
            device):
        super(Transformer, self).__init__()
    
    def forward(self, x):
        pass
