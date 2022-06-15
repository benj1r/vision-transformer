import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from encoder import Encoder


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size,
            channels,
            patch_size,
            classes,
            embed_size,
            heads,
            dropout,
            fwd_expansion,
            layers,
            device):
        super(VisionTransformer, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = dropout
        self.fwd_expansion = fwd_expansion
        self.layers = layers
        self.device = device
            
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        patch_dim = channels * (patch_size**2)
        
        self.encoder = Encoder(
                self.embed_size,
                self.heads,
                self.dropout,
                self.fwd_expansion,
                self.layers
                )
        
        self.cls = nn.Parameter(torch.randn(1,1,embed_size))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_size))
        self.patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, embed_size)
                )
        self.dropout = nn.Dropout(dropout)
        

        self.latent = nn.Identity()

        self.head = nn.Sequential(
                nn.LayerNorm(embed_size),
                nn.Linear(embed_size, classes)
                )
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape
        
        cls = repeat(self.cls, '1 n d -> b n d', b = b)
        x = torch.cat((cls, x), dim=1)
        x += self.position_embedding[:, :(n + 1)]
        
        x = self.dropout(x)
        
        m = None
        x = self.encoder(x, m)
        
        x = x[:,0]
        x = self.latent(x)
        x = self.head(x)
        x = self.softmax(x)

        return x
