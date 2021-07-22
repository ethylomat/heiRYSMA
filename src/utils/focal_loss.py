import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    """Focal loss of binary class, implementation inspired by: https://amaarora.github.io/2020/06/29/FocalLoss.html
    Args:
        pos_weight: A weight of positive samples. Must be a vector with a equal number of classes
        gamma: More the value of γ, more importance will be given to misclassified examples and very less loss will be propagated from easy examples.
        reduction:  (string, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                        'none': no reduction will be applied,
                                        'mean': the sum of the output will be divided by the number of elements in the output,
                                        'sum': the output will be summed.
                                        Note: size_average and reduce are in the process of being deprecated, and in the meantime,
                                        specifying either of those two args will override reduction.
                                        Default: 'mean'
    Returns:
        Loss tensor according to arg reduction
    """
    def __init__(self, pos_weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(pos_weight,reduction=reduction)
        self.gamma = gamma
        self.pos_weight = pos_weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction, pos_weight=self.pos_weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss