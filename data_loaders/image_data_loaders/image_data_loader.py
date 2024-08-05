from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IMAGE_FORMATS = "png"  # Define image formats; replace as needed


# class ImageDataLoader(Dataset):
#     def __init__(self, base_path: str, convert_type: Optional[str] = None, images_format: str = IMAGE_FORMATS):
#         """
#         PyTorch Dataset for loading and preprocessing images.
        
#         Args:
#             base_path (str): Base directory containing image files.
#             channels (List[int]): List of selected channels to extract (e.g., [0] for the first channel).
#             convert_type (Optional[str]): Conversion type, such as "float32".
#             images_format (str): Image file extension (e.g., "jpg").
#         """
#         self.base_path = base_path
#         # self.channels = channels
#         self.convert_type = convert_type
#         self.images_format = images_format

#         # Get list of image file paths
#         self.file_paths = list(map(str, Path(self.base_path).glob(f"*.{self.images_format}")))[:39210]  # Adjust limit as needed

#         # Labels (0 for all images, can be extended for classification tasks)
#         self.labels = [0] * len(self.file_paths)

#         # Transform pipeline (resize, channel selection, type conversion)
#         self.transform = transforms.Compose([
#             transforms.Resize((32, 32)),  # Resize to 256x256
#             transforms.Lambda(self._select_channels),  # Select specific channels
#             # transforms.ToTensor(),  # Convert to tensor
#             transforms.Normalize(mean=[0.5], std=[0.5]),
#         ])

#     # def _select_channels(self, img):
#     #     if not isinstance(img, torch.Tensor):
#     #         # Convert PIL image or other formats to NumPy array
#     #         img_array = np.array(img)
#     #         # Convert to torch tensor with an explicit dtype
#     #         img_array = torch.tensor(img_array, dtype=torch.float32)
#     #     else:
#     #         img_array = img

#     #     # Select specified channels (example for channel 0)
#     #     # if len(img_array.shape) == 3:  # Ensure the image has multiple channels
#     #     #     img_array = img_array[:, :, 0]  # Example: selecting the first channel
#     #     return img_array
    
#     def _select_channels(self, img):
#         if not isinstance(img, torch.Tensor):
#             # Convert PIL image or other formats to NumPy array
#             img_array = np.array(img)
#             # Convert to torch tensor with an explicit dtype
#             img_array = torch.tensor(img_array, dtype=torch.float32)
#         else:
#             img_array = img

#         # Check the shape and permute dimensions to match the desired format
#         # if len(img_array.shape) == 4:  # Assuming input shape is (10, 256, 256, 3)
#         img_array = img_array.permute(2, 0, 1)  # Reorder dimensions to (10, 3, 256, 256)
        
#         return img_array


#     def __len__(self):
#         """
#         Returns the number of images in the dataset.
#         """
#         return len(self.file_paths)

#     def __getitem__(self, index):
#         """
#         Fetch an image and its label.

#         Args:
#             index (int): Index of the image.

#         Returns:
#             torch.Tensor: Processed image tensor.
#             int: Label (default is 0).
#         """
#         file_path = self.file_paths[index]
#         label = self.labels[index]

#         # Load image
#         img = Image.open(file_path).convert("RGB")
#         # print(type(img))# Load and ensure it's in RGB format
#         img = self.transform(img)

#         # Optional conversion type
#         if self.convert_type == "float32":
#             img = img.float()

#         return img, label


# def get_data_loader(base_path: str, convert_type: Optional[str] = None, images_format: str = IMAGE_FORMATS, batch_size: int = 32, num_workers: int = 4):
#     """
#     Create a PyTorch DataLoader for loading images.

#     Args:
#         base_path (str): Base directory containing image files.
#         channels (List[int]): List of selected channels to extract (e.g., [0] for the first channel).
#         convert_type (Optional[str]): Conversion type, such as "float32".
#         images_format (str): Image file extension (e.g., "jpg").
#         batch_size (int): Batch size for loading data.
#         num_workers (int): Number of worker threads for data loading.

#     Returns:
#         torch.utils.data.DataLoader: A DataLoader for loading images.
#     """
#     dataset = ImageDataLoader(base_path, convert_type, images_format)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)




class ImageDataLoader(Dataset):
    def __init__(self, base_path: str, convert_type: Optional[str] = None, images_format: str = IMAGE_FORMATS):
        """
        PyTorch Dataset for loading and preprocessing images.
        
        Args:
            base_path (str): Base directory containing image files.
            convert_type (Optional[str]): Conversion type, such as "float32".
            images_format (str): Image file extension (e.g., "jpg").
        """
        self.base_path = base_path
        self.convert_type = convert_type
        self.images_format = images_format

        # Get list of image file paths
        self.file_paths = list(map(str, Path(self.base_path).glob(f"*.{self.images_format}")))[:39210]

        # Labels (0 for all images, can be extended for classification tasks)
        self.labels = [0] * len(self.file_paths)

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32
            transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        ])

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, index):
        """
        Fetch an image and its label.

        Args:
            index (int): Index of the image.

        Returns:
            torch.Tensor: Processed image tensor.
            int: Label (default is 0).
        """
        file_path = self.file_paths[index]
        label = self.labels[index]

        # Load and preprocess the image
        img = Image.open(file_path).convert("RGB")  # Ensure RGB format
        img = self.transform(img)

        # Optional conversion type
        if self.convert_type == "float32":
            img = img.float()

        return img, label
