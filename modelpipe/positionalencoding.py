import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PositionalEncodings(nn.Module):
    def __init__(self, seq_len=5000, emb_dim=768):
        super(PositionalEncodings, self).__init__()
       
        pe = torch.zeros(seq_len, emb_dim)
        emb_idx = torch.arange(0, emb_dim//2, dtype=torch.float32)
        div_terms = torch.exp(-(2*emb_idx/emb_dim) * math.log(10000))
        pos = torch.arange(seq_len).unsqueeze(1)

        pe[:, 0:emb_dim:2] = torch.sin(pos * div_terms)
        pe[:, 1:emb_dim:2] = torch.cos(pos * div_terms)

        self.register_buffer("pe",pe)

    def encoding_img(self):
        plt.imshow(self.pe.T, cmap='BuGn', aspect='auto')
        plt.tight_layout()
        plt.title("Sinusoidal Positional Encodings")
        plt.xlabel("Token IDs")
        plt.ylabel("Encoding Index")
        plt.show()

    def forward(self, x):
        assert x.size(1) <= self.pe.size(0), "Sequence length exceeds max_seq length"
        return x + self.pe[:x.size(1)].unsqueeze(0)