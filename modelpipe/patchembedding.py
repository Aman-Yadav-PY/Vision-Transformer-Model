import torch
import torch.nn as nn

class PatchEmbeddings(nn.Module):
    def __init__(self, in_feat=1280, out_feat=784, kernel=1, stride=1):
        super(PatchEmbeddings, self).__init__()
        self.patch_conv = nn.Conv2d(in_feat, out_feat, kernel_size=kernel, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_feat)
        self.norm = nn.LayerNorm(out_feat)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_feat))

    def forward(self, x):
        out = self.patch_conv(x)
        out = self.batch_norm(out)
        b, c, h, w = out.size()
        cls_token = self.cls_token.expand(b, -1, -1) 
        out =  out.reshape(b, c, h*w).permute(0, 2, 1)
        out = torch.cat((cls_token, out), dim=1)
        out = self.norm(out)

        return out
        