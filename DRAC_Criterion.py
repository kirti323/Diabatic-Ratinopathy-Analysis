import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Computes the Dice Loss for multi-class image segmentation.
        
        Args:
            logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            
        Returns:
            dice_loss: the Sørensen-Dice loss.
        """
        # Convert targets to one-hot encoding
        targets_one_hot = nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2)
        targets_one_hot = targets_one_hot.float()
        
        # Compute softmax probabilities
        probs = nn.functional.softmax(logits, dim=1)
        
        # Compute Dice coefficient for each class
        dims = (2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_coefficient = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Compute mean Dice loss across classes
        dice_loss = 1. - torch.mean(dice_coefficient)
        
        return dice_loss
