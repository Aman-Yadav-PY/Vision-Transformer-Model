import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class PretrainedBaseModel(nn.Module):
    def __init__(self, pretrained_model:nn.Module=None, freeze_model=True):
        super(PretrainedBaseModel, self).__init__()
        self.pt_model = pretrained_model or efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        if freeze_model:
            self.freeze();

        self.model = self.pt_model.features

        
    def freeze(self):
        for params in self.pt_model.parameters():
            params.requires_grad = False
        self.pt_model.eval() #to prevent batchnorm running -stats(mean & std) update (?)

    def forward(self, x):
        return self.model(x)


model = PretrainedBaseModel()