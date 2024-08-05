import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLPModel(nn.Module):
    def __init__(self):
        super(SimpleMLPModel, self).__init__()

        # Encoder: Image R channel processing
        self.image_encoder = nn.Sequential(
            nn.Linear(32 * 32, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # Encoder: Watermark processing
        self.watermark_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        # Combine image and watermark embeddings
        self.combiner = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32 * 32),
            nn.Tanh(),  # Output normalized between -1 and 1
        )

        # Watermark extraction network
        self.watermark_extractor = nn.Sequential(
            nn.Linear(32 * 32, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Sigmoid(),  # Output normalized between 0 and 1
        )

    def forward(self, image_input, watermark_input):
        # Extract the red channel and flatten
        image_r = image_input[:, 0, :, :].view(-1, 32 * 32)

        # Keep the original green and blue channels
        image_gb = image_input[:, 1:, :, :]  # G and B channels remain unchanged

        # Encode the image
        encoded_image = self.image_encoder(image_r)

        # Encode the watermark
        encoded_watermark = self.watermark_encoder(watermark_input)

        # Combine both embeddings
        combined = torch.cat([encoded_image, encoded_watermark], dim=1)
        watermarked_red_channel = self.combiner(combined).view(-1, 1, 32, 32)

        # Reconstruct the watermarked image with R, G, and B channels
        watermarked_image = torch.cat([watermarked_red_channel, image_gb], dim=1)

        # Extract the watermark from the watermarked image
        extracted_watermark = self.watermark_extractor(
            watermarked_image[:, 0, :, :].view(-1, 32 * 32)  # Use the modified red channel for extraction
        )

        return watermarked_image, extracted_watermark
