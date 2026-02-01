import torch
import torch.nn as nn
from modelpipe import PretrainedBaseModel, PatchEmbeddings, PositionalEncodings,\
                    TransformerEncoder, ClassifierHead


class HybridVitNet(nn.Module):
    def __init__(self, 
                 base_model=None, 
                 pretrained_model=None, 
                 patchifier=None, 
                 pos_encoder=None, 
                 encoder=None,
                 classifier=None,
                 num_classes=None,
                 num_layers=10,
                 nheads=4, 
                 emb_dim=784, 
                 in_channels=1280, 
                 kernel=1, 
                 stride=1, 
                 max_seq_len=5000,
                 **kwargs):
        

        super(HybridVitNet, self).__init__()

        if num_classes: self.classification=True

        self.emb_dim = emb_dim 
        self.base_model = base_model or PretrainedBaseModel(pretrained_model, True)

        self.patchifier = patchifier or PatchEmbeddings(kernel=kernel, stride=stride)

        self.pos_encoder = pos_encoder or PositionalEncodings(seq_len=max_seq_len, emb_dim=emb_dim)

        self.encoder = encoder or TransformerEncoder(num_layers=num_layers,nheads=nheads, emb_dim=emb_dim)
        
        self.classifier = classifier or ClassifierHead(in_feat=emb_dim, out_feat=num_classes)
        self.cls_token =  nn.Parameter(torch.zeros(1, 1, self.emb_dim))

    def forward(self, x):
        x = self.base_model(x)
       
        x = self.patchifier(x)

        if self.classification:
            b, t, d = x.shape
            cls_token = self.cls_token.expand(b, -1, -1)
            x = torch.concat((cls_token, x), dim=1)

        x = self.pos_encoder(x)
        x = self.encoder(x)

        if self.classification:
            x = x[:, 0, :]
            x = self.classifier(x)
            

        return x


        
