import torch
import torch.nn as nn
from attacks.base_attack import BaseAttack

class GaussianNoiseAttack(BaseAttack):
    def __init__(self, mean=0.0, std=0.1):
        """
        Initialize the Gaussian Noise Attack with mean and standard deviation.

        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        super(GaussianNoiseAttack, self).__init__()
        self.mean = mean
        self.std = std

    def gaussian_noise(self, inputs):
        """
        Apply Gaussian noise to the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Noisy tensor.
        """
        noise = torch.normal(mean=self.mean, std=self.std, size=inputs.size(), device=inputs.device)
        return inputs + noise

    def forward(self, inputs):
        """
        Forward pass of the attack.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying Gaussian noise.
        """
        return self.gaussian_noise(inputs)


def gaussian_noise_function(x):
    """
    Functional interface for applying Gaussian Noise Attack.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Noisy tensor.
    """
    return GaussianNoiseAttack()(x)
