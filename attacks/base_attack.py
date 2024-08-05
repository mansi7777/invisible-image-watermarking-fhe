import torch
import torch.nn as nn

class BaseAttack(nn.Module):
    def __init__(self):
        super(BaseAttack, self).__init__()

    def forward(self, inputs):
        """
        Override this method in subclasses to implement specific attack logic.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
