import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss

class FocalDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super(FocalDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(mode='multiclass')
        #self.focal = FocalLoss(mode="multiclass")
        self.focal = SoftCrossEntropyLoss(smooth_factor=0.1)

    def forward(self, output, target):
        focal_loss = self.focal(output, target)
        dice_loss = self.dice(output, target)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

