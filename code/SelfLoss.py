import torch
import torch.nn as nn


class SimilarityMarginLoss(nn.Module):
    def __init__(self):
        super(SimilarityMarginLoss, self).__init__()

    def forward(self, predict, label):
        return torch.clamp(predict - label, min=0).mean()
