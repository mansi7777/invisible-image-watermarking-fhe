import sys
sys.path.append('data_loaders')


from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from attack_id_data_loader.attack_id_data_loader import AttackIdDataset
from image_data_loaders.image_data_loader import ImageDataLoader
from watermark_data_loaders.watermark_data_loader import WatermarkDataLoader
from data_loaders.configs import PREFETCH


class MergedDataset(Dataset):
    def __init__(self, image_dataset: Dataset, watermark_dataset: Dataset, attack_id_dataset: Dataset):
        """
        A dataset that merges image, watermark, and attack ID datasets.

        Args:
            image_dataset (Dataset): Dataset for images.
            watermark_dataset (Dataset): Dataset for watermarks.
            attack_id_dataset (Dataset): Dataset for attack IDs.
        """
        self.image_dataset = image_dataset
        self.watermark_dataset = watermark_dataset
        self.attack_id_dataset = attack_id_dataset

    def __len__(self):
        # Assume all datasets have the same length
        return min(len(self.image_dataset), len(self.watermark_dataset))

    def __getitem__(self, idx):
        image , _= self.image_dataset[idx]
        watermark = self.watermark_dataset[idx]
        attack_id = self.attack_id_dataset[idx]
        # Input: (image, watermark, attack_id), Output: (image, watermark)
        return (image, watermark, attack_id), (image, watermark)


class MergedDataLoader:
    def __init__(self, image_base_path: str, image_convert_type, watermark_size: Tuple[int],
                 attack_min_id: int, attack_max_id: int, batch_size: int, prefetch=PREFETCH):
        """
        A DataLoader that combines image, watermark, and attack ID loaders.

        Args:
            image_base_path (str): Base path for image data.
            image_channels (List[int]): Channels for image data.
            image_convert_type (Any): Type conversion for image data.
            watermark_size (Tuple[int]): Shape of the watermark data.
            attack_min_id (int): Minimum attack ID value.
            attack_max_id (int): Maximum attack ID value.
            batch_size (int): Batch size for the loader.
            prefetch (int): Number of batches to prefetch.
        """
        self.image_data_loader = ImageDataLoader(
            base_path=image_base_path,
            convert_type=image_convert_type
        )
        self.watermark_data_loader = WatermarkDataLoader(
            watermark_size=watermark_size
        )
        self.attack_id_data_loader = AttackIdDataset(
            min_value=attack_min_id,
            max_value=attack_max_id,
            dataset_size=len(self.image_data_loader)
        )
        self.batch_size = batch_size
        self.prefetch = prefetch

    def get_data_loader(self):
        # Instantiate individual datasets
        image_dataset = self.image_data_loader
        watermark_dataset = self.watermark_data_loader
        attack_id_dataset = self.attack_id_data_loader

        # Create merged dataset
        merged_dataset = MergedDataset(image_dataset, watermark_dataset, attack_id_dataset)

        # Create DataLoader with batching and prefetching
        return DataLoader(
            merged_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Adjust if necessary
            prefetch_factor=None,
            num_workers=0  # Adjust based on hardware
        )


# # Example Configuration
# if __name__ == "__main__":
#     from configs import SEED, IMAGE_FORMATS
#     # Example usage
#     image_base_path = "D:/ML/zama/watermark/train_images/train2014"
#     image_channels = [0]  # Example channel selection
#     image_convert_type = torch.float32
#     watermark_size = (16 * 16,)
#     attack_min_id = 0
#     attack_max_id = 4
#     batch_size = 10

#     merged_loader = MergedDataLoader(
#         image_base_path=image_base_path,
#         image_channels=image_channels,
#         image_convert_type=image_convert_type,
#         watermark_size=watermark_size,
#         attack_min_id=attack_min_id,
#         attack_max_id=attack_max_id,
#         batch_size=batch_size
#     )

#     data_loader = merged_loader.get_data_loader()

#     for batch_idx, ((input_image, input_watermark, input_attack_id), (output_image, output_watermark)) in enumerate(data_loader):
#         print(f"Batch {batch_idx + 1}:")
#         print(f"Input: Image Shape: {input_image.shape}, Watermark Shape: {input_watermark.shape}, Attack ID: {input_attack_id}")
#         print(f"Output: Image Shape: {output_image.shape}, Watermark Shape: {output_watermark.shape}")
#         if batch_idx >= 2:  # Example: Stop after 3 batches
#             break
