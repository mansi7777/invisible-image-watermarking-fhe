import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack


class DropOutAttack(BaseAttack):
    def __init__(self, drop_probability=0.3):
        """
        Initialize the DropOutAttack with the specified drop probability.

        Args:
            drop_probability (float): Probability of dropping out elements. Default is 0.3.
        """
        super(DropOutAttack, self).__init__()
        self.drop_probability = drop_probability

    def drop_out(self, inputs):
        """
        Apply dropout to the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying dropout.
        """
        # Generate random mask
        mask_select = torch.rand_like(inputs)
        mask_noise = (mask_select > self.drop_probability).float()

        # Apply the mask to the inputs
        return inputs * mask_noise

    def forward(self, inputs):
        """
        Forward pass of the attack.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying dropout.
        """
        return self.drop_out(inputs)


def drop_out_function(x, drop_probability=0.3):
    """
    Functional interface for applying DropOutAttack.

    Args:
        x (torch.Tensor): Input tensor.
        drop_probability (float): Probability of dropping out elements.

    Returns:
        torch.Tensor: Tensor after applying dropout.
    """
    return DropOutAttack(drop_probability=drop_probability)(x)
