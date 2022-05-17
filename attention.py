import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_size,
            heads):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.D = embed_size // heads

        assert self.D * heads == embed_size, "embed_size needs to be divisible by heads"

        # values, keys, queries definitions
        self.v = nn.Linear(self.D, self.D, bias=False)
        self.k = nn.Linear(self.D, self.D, bias=False)
        self.q = nn.Linear(self.D, self.D, bias=False)

        self.fc = nn.Linear(heads * self.D, embed_size)

    def scaled_dot_attention(self, q, k, m):
        # matmul
        attn = torch.einsum('nqhd,nkhd->nhqk', [q, k])
        
        # scale
        attn = (attn / self.embed_size ** (1/2))
        
        # mask
        if m is not None:
            attn = attn.masked_fill(mask==0, -1e20)
        
        # softmax
        attn = torch.softmax(attn, dim=3)
        
        return attn
        

    def forward(self, v, k, q, m):
        N = q.shape[0]
        
        # values, keys, queries dimensions
        v_dim, k_dim, q_dim = v.shape[1], k.shape[1], q.shape[1]
        v = v.reshape(N, v_dim, self.heads, self.D)
        k = k.reshape(N, k_dim, self.heads, self.D)
        q = q.reshape(N, q_dim, self.heads, self.D)
        
        # linear projection
        v = self.v(v)
        k = self.k(k)
        q = self.q(q)
        
        # attention
        attn = self.scaled_dot_attention(q, k, m)
        
        # matmul
        attn = torch.einsum('nhql,nlhd->nqhd',[attn, v])
        
        # conat
        out = attn.reshape(N, q_dim, self.heads * self.D)
        
        return self.fc(out)
