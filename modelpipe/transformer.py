import torch
import torch.nn as nn
from modelpipe import PatchEmbeddings
from modelpipe import PositionalEncodings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, nhead, d_model):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % nhead == 0

        self.nhead = nhead

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.head_dim  = d_model//nhead

    def attention(self, Q, K, V):
        B, N, D = Q.shape
        Q = Q.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        out = Q @ K.transpose(-2, -1)
        out = torch.softmax(out/(self.head_dim ** (0.5)),dim=-1)
        out = torch.matmul(out, V) #(B, H, N, HD) H-->No. of heads, HD-->Head dims
        out = out.transpose(1, 2) #(B, H, N, HD)  --> (B, N, H, HD)
        return out.contiguous().view(B, N, D)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        attn = self.Wo(self.attention(Q, K, V))

        return self.norm(x+attn)



class TransformerEncoderLayer(nn.Module):
    def __init__(self, nhead, emb_dim, in_feat, kernel, stride, max_seq_len):
        super(TransformerEncoderLayer, self).__init__()
        self.emb_dim = emb_dim
        self.patchs = PatchEmbeddings(in_feat=in_feat, out_feat=emb_dim, \
                                      kernel=kernel, stride = stride)
        self.position_encoding = PositionalEncodings(seq_len=max_seq_len, emb_dim=emb_dim)
        self.mhsa = MultiHeadSelfAttention(nhead=nhead, d_model=emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

        self.model = nn.Sequential(nn.Linear(emb_dim, emb_dim*4), nn.GELU(), 
                                   nn.Linear(emb_dim*4, emb_dim))

    def forward(self, x):
        x = self.patchs(x)
        x = self.position_encoding(x)
        x = self.mhsa(x)
        return self.norm(x + self.model(x))
        


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()