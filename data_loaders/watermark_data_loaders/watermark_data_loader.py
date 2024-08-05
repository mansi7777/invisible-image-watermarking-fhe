import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

SEED = 42  # Default seed; replace with your seed or import from a config file


class WatermarkDataLoader(Dataset):
    def __init__(self, watermark_size, seed=SEED):
        """
        PyTorch Dataset for generating random watermarks.

        Args:
            watermark_size (tuple): Shape of the watermark to generate.
            seed (int): Random seed for reproducibility.
        """
        super(WatermarkDataLoader, self).__init__()
        self.watermark_size = watermark_size
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)

    def __len__(self):
        """
        Return an arbitrary large number since this dataset is infinite in nature.
        """
        return int(1e9)  # Simulating an effectively infinite dataset

    def __getitem__(self, index):
        """
        Generate a single watermark sample.

        Args:
            index (int): Index (not used since data is generated on-the-fly).

        Returns:
            torch.Tensor: A randomly generated watermark tensor.
        """
        watermark = torch.round(
            torch.rand(self.watermark_size, generator=self.generator, dtype=torch.float32)
        )
        return watermark


def get_data_loader(watermark_size, seed=SEED, batch_size=32, num_workers=0):
    """
    Create a PyTorch DataLoader for the WatermarkDataLoader.

    Args:
        watermark_size (tuple): Shape of the watermark to generate.
        seed (int): Random seed for reproducibility.
        batch_size (int): Batch size for loading data.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for generating watermarks.
    """
    dataset = WatermarkDataLoader(watermark_size, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
