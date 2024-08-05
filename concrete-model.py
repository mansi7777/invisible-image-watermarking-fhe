# from models.wavetf_model import WaveTFModel
from models.wavetf_small import SimpleMLPModel
from models.quant_wavetf import QuantizedMLPModel
from models.wavetf_32 import WaveTF32x32
from models.quant_wave32 import QuantWaveTF32x32
from data_loaders.merged_data_loader import MergedDataLoader
from typing import Dict, Callable
from collections import OrderedDict
from torch.utils.data.dataloader import DataLoader
from concrete.ml.torch.compile import compile_brevitas_qat_model
from brevitas import config
import torch
import torch.nn as nn
import torch.optim as optim
from configs import *
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from concrete.fhe import Configuration
from concrete.fhe import Exactness
import os
import torchvision.utils as vutils

def fhe_compatibility(model: Callable, data: DataLoader, device: str) -> Callable:
    """Test if the model is FHE-compatible.

    Args:
        model (Callable): The Brevitas model.
        data (DataLoader): The data loader.
        device (str): Specifies the device to run during the compilation, either 'cpu' or 'gpu'.

    Returns:
        Callable: Quantized model.
    """
    tensor1 = data[0][0]
    tensor2 = data[0][1]

    # Convert the tensors to numpy arrays
    numpy_array1 = tensor1.cpu().numpy()
    numpy_array2 = tensor2.cpu().numpy()

    # Create a tuple of the two numpy arrays
    numpy_tuple = (numpy_array1, numpy_array2)
    
    config = Configuration(
        enable_tlu_fusing=True,
        print_tlu_fusing=False,
        enable_unsafe_features=True,
        use_insecure_key_cache=True,
        insecure_key_cache_location="~/.cml_keycache",
        show_progress=True,
        use_gpu=False
    )
    qmodel = compile_brevitas_qat_model(
        model.to("cpu"),
        # Training
        torch_inputset=numpy_tuple,
        show_mlir=False,
        output_onnx_file="test.onnx",
        rounding_threshold_bits={"n_bits": 8, "method": Exactness.APPROXIMATE},
        configuration=config,
        verbose=True,
        device=device,
    )

    return qmodel

# Function to calculate watermark mismatch
def calculate_watermark_mismatch(predicted, ground_truth):
    predicted_binary = (predicted > 0.5).int()  # Convert predictions to binary (threshold at 0.5)
    ground_truth_binary = (ground_truth > 0.5).int()  # Convert ground truth to binary
    mismatches = (predicted_binary != ground_truth_binary).sum().item()  # Count mismatched entries
    return mismatches

# Hyperparameters
lambda1 = IMAGE_LOSS_WEIGHT  # Weight for the image reconstruction loss
lambda2 = WATERMARK_LOSS_WEIGHT  # Weight for the watermark reconstruction loss
learning_rate = 0.001
num_epochs = 60000
device = 'cpu'

if __name__ == "__main__":
    # Paths and configurations
    image_base_path = "train"
    image_convert_type = torch.float32
    watermark_size = (8 * 8,)
    attack_min_id = 0
    attack_max_id = 1
    batch_size = 100

    # Initialize model, loss, optimizer, and data loader
    # model = SimpleMLPModel().to(device)
    # model = QuantWaveTF32x32(image_size = (batch_size, 3, 32, 32),watermark_size=watermark_size).to(device)
    model=QuantizedMLPModel().to(device)
    merged_loader = MergedDataLoader(
        image_base_path=image_base_path,
        image_convert_type=image_convert_type,
        watermark_size=watermark_size,
        attack_min_id=attack_min_id,
        attack_max_id=attack_max_id,    
        batch_size=batch_size
    )
    data_loader = merged_loader.get_data_loader()

    criterion1 = nn.MSELoss()  # L2 loss
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    # Compile the model in FHE mode and also checking the FHE-compatibility and whether homomorphic inference is achievable or not before moving on.
    data_calibration= next(iter(data_loader))
    qmodel = fhe_compatibility(model, data_calibration, device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, ((input_image, input_watermark, input_attack_id), (output_image, output_watermark)) in enumerate(data_loader):
            # Move data to device
            input_image = input_image.to(device)
            input_watermark = input_watermark.to(device)
            input_attack_id = input_attack_id.to(device)
            output_image = output_image.to(device)
            output_watermark = output_watermark.to(device)

            # Forward pass
            wavelet_inverse_image, extracted_watermark = model(input_image, input_watermark)
            
            # Compute losses
            image_loss = criterion1(wavelet_inverse_image[:, :1, :, :], output_image[:, :1, :, :])  # L2 loss for images
            watermark_loss = criterion2(extracted_watermark, output_watermark)  # CrossEntropy loss for watermarks
            loss = lambda1 * image_loss + lambda2 * watermark_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}")

        # End of epoch logging
        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

        # Save model only if it achieves a new lowest loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }, "wavetf_32.pth")
            print(f"New best model saved with loss {best_loss:.4f}")

    print("Training complete. Best model saved.")

    # Paths and configurations
    image_base_path = "train" # Path to validation images
    output_dir = "./test_results"  # Directory to save the output images
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameters (match training setup)
    watermark_size = (8*8,)
    batch_size = 10

    
    checkpoint = torch.load("wavetf_32.pth", map_location=torch.device('cpu'))  # Load checkpoint
    

    # Initialize validation data loader
    merged_loader = MergedDataLoader(
        image_base_path=image_base_path,
        image_convert_type=torch.float32,
        watermark_size=watermark_size,
        attack_min_id=0,
        attack_max_id=1,
        batch_size=batch_size
    )
    data_loader = merged_loader.get_data_loader()

    # Evaluate on a single batch
    with torch.no_grad():
        for batch_idx, ((input_image, input_watermark, input_attack_id), (output_image, output_watermark)) in enumerate(data_loader):
            # Move data to device
            input_image = input_image.to(device)
            input_watermark = input_watermark.to(device)
            input_attack_id = input_attack_id.to(device)
            output_image = output_image.to(device)
            output_watermark = output_watermark.to(device)
#  Using FHE simulation since it is faster than the actual FHE compilation, because it relies only on Python
            fhe_mode = "simulate" 
            wavelet_inverse_image, extracted_watermark = qmodel.forward(input_image.numpy(), input_watermark.numpy(), fhe=fhe_mode)
            wavelet_inverse_image = torch.from_numpy(wavelet_inverse_image)
            extracted_watermark = torch.from_numpy(extracted_watermark)

            # Save predicted vs ground truth images
            for i in range(batch_size):
                # Save ground truth and predicted images
                vutils.save_image(
                    output_image[i],
                    os.path.join(output_dir, f"ground_truth_image_{i + 1}.png"),
                    normalize=True
                )
                vutils.save_image(
                    wavelet_inverse_image[i],
                    os.path.join(output_dir, f"predicted_image_{i + 1}.png"),
                    normalize=True
                )

                # Calculate watermark mismatch
                mismatches = calculate_watermark_mismatch(extracted_watermark[i], output_watermark[i])
                print(f"Sample {i + 1}: {mismatches} / 64 watermark entries are incorrect.")

            print(f"Saved predicted and ground truth images for batch {batch_idx + 1}")
            break  # Process only the first batch

