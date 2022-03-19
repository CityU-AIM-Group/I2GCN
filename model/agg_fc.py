import torch 
import torch.nn as nn
import torch.nn.functional as F 

class Aggregator(torch.nn.Module):
    def __init__(self, class_num=3, in_dim=512, mean_flag=True):
        super(Aggregator, self).__init__()
        self.classifier = nn.Linear(in_dim, class_num, bias=False)
        self.mean_flag = mean_flag

    def forward(self, x):
        # x in [B, C, Z]
        x_shape = x.shape
        if self.mean_flag:
            x = torch.mean(x, dim=-1, keepdim=False).view(x_shape[0], x_shape[1])
        else:
            x, _ = torch.max(x, dim=-1, keepdim=False)
            x = x.view(x_shape[0], x_shape[1])
        output = self.classifier(x)
        return output