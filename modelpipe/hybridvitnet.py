import torch
import torch.nn as nn
from modelpipe import PatchEmbeddings, PositionalEncodings, \
                    MultiHeadSelfAttention, TransformerEncoder,\
                    ClassifierHead