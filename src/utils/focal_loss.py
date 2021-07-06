import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, pos_weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(pos_weight,reduction=reduction)
        self.gamma = gamma
        self.pos_weight = pos_weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction, pos_weight=self.pos_weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss