import torch
import torch.nn as nn
from attacks.base_attack import BaseAttack


class SaltPepperAttack(BaseAttack):
    def __init__(self, prob: float = 0.1, salt_vs_pepper: float = 0.5):
        """
        Initialize the SaltPepperAttack.

        Args:
            prob (float): Probability of an element to be corrupted by noise.
            salt_vs_pepper (float): Proportion of salt noise (vs pepper noise). Default is 0.5.
        """
        super(SaltPepperAttack, self).__init__()
        self.prob = prob
        self.salt_vs_pepper = salt_vs_pepper

    def salt_pepper(self, inputs):
        """
        Apply salt-and-pepper noise to the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Tensor with salt-and-pepper noise added.
        """
        shp = inputs.shape

        # Create a random mask for where to apply noise
        mask_select = (torch.rand(shp, device=inputs.device) < self.prob).float()  # Probability for noise application

        # Create a mask for salt (1) or pepper (0)
        mask_noise = (torch.rand(shp, device=inputs.device) < self.salt_vs_pepper).float()  # Salt vs pepper

        # Apply noise
        noisy_inputs = inputs * (1 - mask_select) + mask_noise * mask_select
        return noisy_inputs

    def forward(self, inputs):
        """
        Forward pass of the attack.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Tensor with salt-and-pepper noise added.
        """
        return self.salt_pepper(inputs)


def salt_pepper_function(x, prob: float = 0.1, salt_vs_pepper: float = 0.5):
    """
    Functional interface for applying SaltPepperAttack.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        prob (float): Probability of an element to be corrupted by noise.
        salt_vs_pepper (float): Proportion of salt noise (vs pepper noise). Default is 0.5.

    Returns:
        torch.Tensor: Tensor with salt-and-pepper noise added.
    """
    return SaltPepperAttack(prob=prob, salt_vs_pepper=salt_vs_pepper)(x)
