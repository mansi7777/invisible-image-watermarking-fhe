from concrete.ml.quantization.quantized_module import QuantizedModule
from concrete.ml.torch.compile import compile_brevitas_qat_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

class QuantizedMLPModel(nn.Module):
    def __init__(self):
        super(QuantizedMLPModel, self).__init__()
        self.quant_inp = qnn.QuantIdentity( act_quant=Int8ActPerTensorFloat )
        # self.quant_inp = qnn.QuantIdentity()

        # Encoder: Image R channel processing
        self.image_encoder = nn.Sequential(
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantLinear(32 * 32, 1024, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(1024, 512, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(512, 256, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(256, 128, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
        )

        # Encoder: Watermark processing
        self.watermark_encoder = nn.Sequential(
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantLinear(64, 128, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
        )

        # Combine image and watermark embeddings
        self.combiner = nn.Sequential(
            qnn.QuantLinear(128 + 128, 256, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(256, 512, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(512, 1024, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(1024, 32 * 32, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            nn.Sigmoid(),  # Output normalized between -1 and 1
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),  
        )

        # Watermark extraction network
        self.watermark_extractor = nn.Sequential(
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantLinear(32 * 32, 1024, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(1024, 512, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(512, 256, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(256, 128, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(inplace=True),
            qnn.QuantLinear(128, 64, weight_quant=Int8WeightPerTensorFloat , return_quant_tensor=True ),
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            nn.Sigmoid(),  # Output normalized between 0 and 1
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat)  
        )

    def forward(self, image, watermark):
        image=self.quant_inp(image)
        watermark=self.quant_inp(watermark)
        # Extract the red channel and flatten
        image_r = self.quant_inp(image[:, 0, :, :].view(-1, 32 * 32))  # Keep dimension
        image_gb = self.quant_inp(image[:, 1:, :, :])
        # Encode the image
        encoded_image = self.image_encoder(image_r)
        encoded_image = self.quant_inp(encoded_image)
        # Encode the watermark
        encoded_watermark = self.watermark_encoder(watermark)
        encoded_watermark = self.quant_inp(encoded_watermark)
        # Combine both embeddings
        combined = torch.cat([encoded_image, encoded_watermark], dim=1)
        watermarked_red_channel = self.combiner(combined).view(-1, 1, 32, 32)
        watermarked_red_channel = self.quant_inp(watermarked_red_channel)

        # Reconstruct the watermarked image with R, G, and B channels
        watermarked_image = torch.cat([watermarked_red_channel, image_gb], dim=1)

        # Extract the watermark from the watermarked image
        extracted_watermark = self.watermark_extractor(
            watermarked_image[:, 0, :, :].view(-1, 32 * 32)  # Use the modified red channel for extraction
        )
        watermarked_image=self.quant_inp(watermarked_image)
        return image, watermark