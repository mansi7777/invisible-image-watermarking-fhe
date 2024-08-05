import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from attacks.base_attack import BaseAttack


class RotationAttack(BaseAttack):
    def __init__(self, angle: float = 90.0):
        """
        Initialize the RotationAttack with the specified angle.

        Args:
            angle (float): Rotation angle in degrees. Default is 90.
        """
        super(RotationAttack, self).__init__()
        self.angle = angle

    def rotation(self, inputs):
        """
        Apply rotation to the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Rotated tensor of the same shape.
        """
        # Rotate each image in the batch
        rotated_images = torch.stack([
            F.rotate(img, self.angle, fill=0.0) for img in inputs
        ])
        return rotated_images

    def forward(self, inputs):
        """
        Forward pass of the attack.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Rotated tensor of the same shape.
        """
        return self.rotation(inputs)


def rotation_function(x, angle: float = 90.0):
    """
    Functional interface for applying RotationAttack.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        angle (float): Rotation angle in degrees. Default is 90.

    Returns:
        torch.Tensor: Rotated tensor.
    """
    return RotationAttack(angle=angle)(x)
