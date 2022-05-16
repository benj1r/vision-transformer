import torch
import torch.nn as nn

from attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            fwd_expansion):
        super(EncoderBlock, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.fwd_expansion = fwd_expansion

        self.MHA = MultiHeadAttention(self.embed_size, self.heads)
        self.norm = nn.LayerNorm(self.embed_size)
        self.ff = nn.Sequential(
                nn.Linear(embed_size, fwd_expansion * embed_size),
                nn.ReLU(),
                nn.Linear(embed_size * fwd_expansion, embed_size)
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, m):
        attention = self.MHA(v, k, q, m)
        x = self.dropout(self.norm(attention + q))
        fwd = self.ff(x)
        out = self.dropout(self.norm(fwd + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            fwd_expansion,
            layers,
            ):
        super(Encoder, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = dropout
        self.fwd_expansion = fwd_expansion
        self.num_layers = layers
        
        self.layers = nn.ModuleList(
                [
                    EncoderBlock(
                        self.embed_size,
                        self.heads,
                        self.dropout,
                        self.fwd_expansion)
                    for _ in range(self.num_layers)
                    ]
                )
        
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, m):
        out = self.dropout(x)
        
        for layer in self.layers:
            out = layer(out, out, out, m)
        return out
