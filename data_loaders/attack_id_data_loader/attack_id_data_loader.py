from torch.utils.data import Dataset, DataLoader
import torch
import random

class AttackIdDataset(Dataset):
    def __init__(self, min_value: int, max_value: int, dataset_size: int):
        """
        Dataset for generating random attack IDs.

        Args:
            min_value (int): Minimum value (inclusive) of the random range.
            max_value (int): Maximum value (exclusive) of the random range.
            dataset_size (int): Number of items in the dataset.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.dataset_size = dataset_size

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return self.dataset_size

    def __getitem__(self, index):
        """
        Fetch a random attack ID.

        Args:
            index (int): Index (not used, but required for Dataset API).

        Returns:
            torch.Tensor: A tensor containing a random integer in the range [min_value, max_value).
        """
        # Generate a random integer in the specified range
        attack_id = random.randint(self.min_value, self.max_value - 1)
        return torch.tensor(attack_id, dtype=torch.int32)


def get_data_loader(min_value: int, max_value: int, batch_size: int, dataset_size: int):
    """
    Creates a PyTorch DataLoader for generating random attack IDs.

    Args:
        min_value (int): Minimum value (inclusive) of the random range.
        max_value (int): Maximum value (exclusive) of the random range.
        batch_size (int): Number of random IDs per batch.
        dataset_size (int): Total number of items in the dataset.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for generating random attack IDs.
    """
    dataset = AttackIdDataset(min_value, max_value, dataset_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# # Example Usage
# if __name__ == "__main__":
#     min_value = 0
#     max_value = 4
#     batch_size = 5
#     dataset_size = 100  # Define a fixed dataset size

#     # Create DataLoader
#     data_loader = get_data_loader(min_value, max_value, batch_size, dataset_size)

#     # Iterate through DataLoader
#     for i, batch in enumerate(data_loader):
#         print(f"Batch {i + 1}: {batch}")
#         if i >= 4:  # Stop after 5 batches
#             break
