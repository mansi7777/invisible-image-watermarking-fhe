import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from typing import Tuple

class WaveTF32x32(nn.Module):
    def __init__(self, image_size: Tuple[int], watermark_size: Tuple[int] = (16 * 16,), wavelet_type='haar'):
        super(WaveTF32x32, self).__init__()
        self.image_size = image_size
        self.watermark_size = watermark_size
        self.wavelet_type = wavelet_type
        self.watermark_x_size = int(watermark_size[0] ** 0.5)

        self.dwt = DWTForward(J=1, wave=self.wavelet_type, mode='zero')
        self.idwt = DWTInverse(wave=self.wavelet_type, mode='zero')

        # Preprocess watermark to match image size
        self.watermark_preprocess = nn.Sequential(
            nn.Linear(self.watermark_size[0], 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
        )

        # Embedding Network
        self.embedding_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Extraction Network
        self.extraction_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Added Average Pooling
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image_input, watermark_input):
        # Extract R, G, B channels
        image_input_r = image_input[:, :1, :, :]
        image_input_gb = image_input[:, 1:, :, :]
        
        # Apply DWT to R channel
        coeffs = self.dwt(image_input_r)
        LL, wavelet_coeffs = coeffs[0], coeffs[1][0]
        wavelet_image = LL / 2

        # Preprocess watermark
        preprocessed_watermark = self.watermark_preprocess(watermark_input)
        preprocessed_watermark = preprocessed_watermark.view(-1, 1, 16, 16)

        # Embed watermark
        concatenated = torch.cat([wavelet_image, preprocessed_watermark], dim=1)
        watermarked_image = self.embedding_layers(concatenated)
        reconstructed_coeffs = (watermarked_image * 2, [wavelet_coeffs])
        wavelet_inverse_r = self.idwt(reconstructed_coeffs)

        # Reconstruct final image with original G and B channels
        wavelet_inverse_image = torch.cat([wavelet_inverse_r, image_input_gb], dim=1)

        # Extract watermark
        attacked_coeffs = self.dwt(wavelet_inverse_r)
        extracted_coeffs = attacked_coeffs[0]
        extracted_watermark = self.extraction_layers(extracted_coeffs)
        extracted_watermark = extracted_watermark.view(-1, self.watermark_size[0])

        return wavelet_inverse_image, extracted_watermark