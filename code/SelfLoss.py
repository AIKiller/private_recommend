import torch
import torch.nn as nn


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, predict, label):
        return torch.clamp(predict - label, min=0).mean()

    # def forward(self, predict, label):
    #     #   label - 20 < x < label
    #     min_loss = torch.clamp((label-0.4) - predict, min=0).mean()
    #     max_loss = torch.clamp(predict - label, min=0)
    #     return (min_loss + max_loss).mean()
