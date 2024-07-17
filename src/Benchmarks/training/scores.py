import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


class DiceCoeff(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCoeff, self).__init__()

    def forward(self, inputs, targets, smooth=1.):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class IoUCoeff(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoUCoeff, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU


class Precision(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Precision, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = ((inputs == 1) & (targets == 1)).sum().float()
        FP = ((inputs == 1) & (targets == 0)).sum().float()

        precision = TP / (TP + FP + 1e-10)  # Avoid division by zero

        return precision


class Recall(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Recall, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = ((inputs == 1) & (targets == 1)).sum().float()
        FN = ((inputs == 0) & (targets == 1)).sum().float()

        recall = TP / (TP + FN + 1e-10)  # Avoid division by zero

        return recall


class Accuracy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Accuracy, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = ((inputs == 1) & (targets == 1)).sum().float()
        TN = ((inputs == 0) & (targets == 0)).sum().float()
        FP = ((inputs == 1) & (targets == 0)).sum().float()
        FN = ((inputs == 0) & (targets == 1)).sum().float()

        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)  # Avoid division by zero

        return accuracy
