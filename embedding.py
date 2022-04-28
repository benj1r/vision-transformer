import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            channels
            ):
        super(PatchEmbedding, self).__init__()

    def forward(self, x):
        pass


class PositionEmbedding(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            pad_idx
            ):
        super(PositionEmbedding, self).__init__()

    def forward(self, x):
        pass

