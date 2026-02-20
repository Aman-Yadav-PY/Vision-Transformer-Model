import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_feat, out_feat, logits=False, **kwargs):
        super(ClassifierHead, self).__init__()

        self.hidden = [in_feat//2**(i+1) for i in range(2)]
        self.hidden = self.hidden + list(reversed(self.hidden))
        self.logits = logits

            
        self.fnn_in = nn.Linear(in_feat, self.hidden[0])
        self.fnn = nn.ModuleList()

        for _in, _out in zip(self.hidden, self.hidden[1:]):
            self.fnn.append(nn.Linear(_in, _out))
            self.fnn.append(nn.GELU())

        self.fnn_out = nn.Linear(self.hidden[-1], out_feat)

        if not logits:
            self.probs = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fnn_in(x)

        for layer in self.fnn:
            x = layer(x)

        x = self.fnn_out(x)
        return x if self.logits else self.probs(x)


        
