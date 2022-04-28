import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            fwd_expansion):
        super(EncoderBlock, self).__init__()
    
    def forward(self, v, k, q, m):
        pass

class Encoder(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            fwd_expansion,
            layers,
            max_len
            ):
        super(Encoder, self).__init__()

    def forward(self, x, m):
        pass

