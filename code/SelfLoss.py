import torch
import torch.nn as nn


class SimilarityMarginLoss(nn.Module):
    def __init__(self):
        super(SimilarityMarginLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, predict, label):
        target = torch.zeros_like(predict)
        target[:] = label
        return self.l1_loss(predict, target)

