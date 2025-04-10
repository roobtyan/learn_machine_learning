import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        self.k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.q = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, k, q, v, mask=None):
        N = q.shape[0]
        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1]

        k = k.reshape(N, k_len, self.heads, self.head_dim)
        q = q.reshape(N, q_len, self.heads, self.head_dim)
        v = v.reshape(N, v_len, self.heads, self.head_dim)

        k = self.k(k)
        q = self.q(q)
        v = self.v(v)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        energy = torch.matmul(q, k)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        v = v.permute(0, 2, 1, 3)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).reshape(N, q_len, self.embed_size)

        out = self.fc_out(out)
        return out