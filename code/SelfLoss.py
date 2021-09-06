import torch
import torch.nn as nn


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, predict, label):
        return torch.clamp(predict - label, min=0).mean()
