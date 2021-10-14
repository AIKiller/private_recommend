import torch
import torch.nn as nn


class SimilarityMarginLoss(nn.Module):
    def __init__(self):
        super(SimilarityMarginLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, predict, label):
        return torch.clamp(predict - label, min=0).mean()

    # def forward(self, predict, label):
    #     target = torch.zeros_like(predict)
    #     target[:] = label
    #     return self.l1_loss(predict, target)

    # def forward(self, predict, label):
    #     #   label - 0.4 < x < label
    #     min_loss = torch.clamp((label-0.4) - predict, min=0).mean()
    #     max_loss = torch.clamp(predict - label, min=0)
    #     return (min_loss + max_loss).mean()
