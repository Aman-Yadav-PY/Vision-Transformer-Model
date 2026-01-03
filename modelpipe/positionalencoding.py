import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


import math

class PositionalEncodings(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super(PositionalEncodings, self).__init__()
        self.dmodel = emb_dim

        pe = torch.zeros(seq_len, self.dmodel)
        emb_idx = torch.arange(0, self.dmodel//2, dtype=torch.float32)
        div_terms = torch.exp(-(2*emb_idx/self.dmodel) * math.log(10000))
        pos = torch.arange(seq_len).unsqueeze(1)

        pe[:, 0:self.dmodel:2] = torch.sin(pos * div_terms)
        pe[:, 1:self.dmodel:2] = torch.cos(pos * div_terms)

        self.register_buffer("pe",pe)

    def encoding_img(self):
        plt.imshow(self.pe.T, cmap='BuGn', aspect='auto')
        plt.tight_layout()
        plt.title("Sinusoidal Positional Encodings")
        plt.xlabel("Position")
        plt.ylabel("Encoding Index")
        plt.show()

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)