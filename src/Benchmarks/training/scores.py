import torch
import torch.nn as nn


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


class DiceLossMulticlass(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLossMulticlass, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: Tensor of shape (batch_size, num_classes, H, W)
        targets: Tensor of shape (batch_size, H, W) with class indices
        """
        num_classes = inputs.shape[1]

        # Apply softmax to get class probabilities
        inputs = torch.softmax(inputs, dim=1)
        targets = targets.to(inputs.device)

        # Create a one-hot encoding of targets
        targets_one_hot = torch.eye(num_classes, device=inputs.device)[targets].permute(0, 3, 1, 2)

        # Flatten tensors for each class
        inputs_flat = inputs.reshape(num_classes, -1)
        targets_flat = targets_one_hot.reshape(num_classes, -1)

        # Calculate Dice for each class
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)

        # Return mean Dice Loss over all classes
        dice_loss = 1 - dice.mean()

        return dice_loss
