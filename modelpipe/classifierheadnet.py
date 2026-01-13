import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_feat, out_feat, **kwargs):
        super(ClassifierHead, self).__init__()

        if kwargs:
            self.hidden = kwargs['hidden']
            
        self.fnn_in = nn.Linear(in_feat, self.hidden[0])
        self.fnn = nn.ModuleList()

        for _in, _out in zip(self.hidden, self.hidden[1:]):
            self.fnn.append(nn.Linear(_in, _out))
            self.fnn.append(nn.GELU())

        self.fnn_out = nn.Linear(self.hidden[-1], out_feat)
        self.probs = nn.Softmax()

    def forward(self, x):
        x = self.fnn_in(x)

        for layer in self.fnn:
            x = layer(x)

        x = self.fnn_out(x)
        return self.probs(x)


        