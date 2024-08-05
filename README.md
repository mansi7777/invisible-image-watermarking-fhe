# Privacy-Preserving Invisible Image Watermarking System using Concrete ML

## Overview 
Invisible image watermarking is a technique used to embed hidden information within digital images without visibly altering their appearance. This challenge focuses on implementing this technology using Fully Homomorphic Encryption (FHE), specifically through Concrete ML, to enhance privacy and security in the watermarking process.
<br>
## Features:
- **Fully Homomorphic Encryption (FHE)**: Ensures that the model can perform inference on encrypted data without the need for decryption, preserving data confidentiality.
- **Quantization Aware Training (QAT)**: Trains models that are compatible with FHE, ensuring efficient and accurate inference.
- **Watermark Detection**: If more than 75% of the bits match with the watermark, we can say with reasonable certainty that the image is watermarked.
- **Fast Responses**: The system is designed to provide fast responses during both training and inference.
- **Indistinguishable Watermarked Images**: The watermarked image is indistinguishable from the original image to the naked eye.
## Methodology
The system employs Quantization Aware Training (QAT) to train models that are compatible with FHE. By leveraging Concrete ML, the models can perform inference on encrypted data without the need for decryption, preserving data confidentiality throughout the process.
<br>
### Architecture:
Our architecture and learning startegy is inspired by the paper ["Convolutional Neural Network-Based Image Watermarking using Discrete Wavelet Transform"](https://arxiv.org/abs/2210.06179)
<li><b>QuantizedMLPModel:</b>
  
![QuantWaveTFsmall](https://github.com/aaravm/watermark/blob/master/Zama_small.png?raw=true)
<ul>
<li>
<b>Input Quantization:</b>
Both the full-color image (assumed to be of shape [batch, 3, 32, 32]) and the watermark (a 64-dimensional vector) are first passed through a quantization layer (qnn.QuantIdentity) to ensure low-precision (Int8) processing.
</li>
<li>
<b>Red Channel Extraction and Encoding:</b>
The model extracts the red channel from the image by taking the first channel and reshaping it from [batch, 32, 32] to a flattened [batch, 1024] vector. This flattened red channel is then fed through the Image Encoder, a sequence of quantized linear layers interleaved with quantized ReLU activations, ultimately encoding the red channel into a 128-dimensional representation.
</li>
<li>
<b>Watermark Encoding:</b>
In parallel, the watermark vector is processed by the Watermark Encoder. This encoder applies a quantized identity, followed by a quantized linear layer that maps the 64-dimensional input to 128 dimensions and a ReLU activation. The result is a 128-dimensional encoded watermark.
</li>
<li>
<b>Embedding via the Combiner:</b>
The two 128-dimensional outputs (one from the image encoder and one from the watermark encoder) are concatenated along the feature dimension to form a 256-dimensional joint representation. This combined vector is then passed through the Combiner network—a series of quantized linear layers with ReLU activations—that transforms it into a 1024-dimensional vector. This vector is reshaped into a 1-channel image of size [batch, 1, 32, 32], which represents the watermarked red channel. A final Sigmoid activation ensures the output values are normalized.
</li>
<li>
<b>Image Reconstruction and Watermark Extraction:</b>
The modified red channel is concatenated with the original green-blue channels (which were preserved earlier) to reconstruct the full watermarked image. Additionally, for watermark extraction, the modified red channel is flattened back into a 1024-dimensional vector and processed through the Watermark Extractor—another series of quantized linear layers with ReLU and a concluding Sigmoid activation—to recover a 64-dimensional watermark. Although the extractor computes this output, the model’s forward method ultimately returns the original image and watermark (leaving it to the training setup to compute losses comparing these with the watermarked image and the extracted watermark).
</li>
</ul>
<li><b>QuantWaveTF32x32 :</b>

![QuantWaveTF32x32](https://github.com/aaravm/watermark/blob/master/Zama_wavetf32.png?raw=true)
<ul>
<li><b>Preprocessing:</b> The model first quantizes both the input image and watermark. The image’s red channel is isolated and decomposed via a one-level DWT into approximation (LL) and detail coefficients. The LL component is scaled (divided by 2) to serve as a base for embedding.</li>
<li><b>Watermark Processing:</b> The watermark (a vector of size 16×16 or 256 elements) is passed through a series of quantized linear layers, activated by ReLU and Sigmoid functions, to produce a feature map reshaped to (batch, 1, 16, 16).</li>
<li><b>Embedding:</b> The preprocessed watermark and the scaled LL component are concatenated along the channel dimension (forming a 2-channel input) and processed by the embedding network’s convolutional layers. The network outputs a modified LL coefficient map (after a Tanh activation and subsequent scaling by 2) that, when combined with the unchanged detail coefficients, is passed through the inverse DWT to reconstruct a modified red channel.</li>
<li><b>Reconstruction & Extraction:</b> The final watermarked image is reassembled by concatenating the reconstructed red channel with the original green and blue channels. For watermark extraction, the red channel of the watermarked image is decomposed using DWT again, and its LL component is fed through the extraction network (a lightweight CNN with an average pooling layer) to recover the watermark, which is then reshaped back into its original vector form.
</li>
</ul>

### Loss Function
The two-term loss function is used for training the network. The similarity between a watermarked image and
the host image is measured by the mean square error (MSE) loss function shown in Equation 1: <br>
![image](https://github.com/user-attachments/assets/fc99edca-0424-450e-a719-d1bb686923f7)

Also, the mean absolute error (MAE) function is used as the loss function between the extracted watermark and input watermark displayed in Equation 2: <br>
![image](https://github.com/user-attachments/assets/e0df4b4f-7f38-497d-848e-1ffd689df469)

As shown in Equation 3, a combination of these functions determines the loss function of the whole network <br>
![image](https://github.com/user-attachments/assets/1836c297-950f-4daf-a41c-8f5c9d3e74d7)

### Quantization and using FHE in concrete-ml
1. **Quantizing the torch model:** The model [QuantWaveTF32x32](https://github.com/aaravm/watermark/blob/master/models/quant_wave32.py), [QuantizedMLPModel](https://github.com/aaravm/watermark/blob/master/models/quant_wavetf.py#L9) is a quantized version made by quantizing [WaveTF32x32](https://github.com/aaravm/watermark/blob/master/models/wavetf_32.py), [SimpleMLPModel](https://github.com/aaravm/watermark/blob/master/models/wavetf_small.py#L5) by replacing nn.Linear by qnn.QuantLinear, nn.ReLU by qnn.QuantReLU and using QuantIdentity to quantize inputs and outputs with the parameter Int8WeightPerTensorFloat for 8-bit signed integer weight quantization with a per-tensor floating-point scale factor
2. **Converting our Quantization Aware Training (QAT) model into an FHE-compatible quantized module:** The [QuantWaveTF32x32](https://github.com/aaravm/watermark/blob/master/models/quant_wave32.py), [QuantizedMLPModel](https://github.com/aaravm/watermark/blob/master/models/quant_wavetf.py#L9) is converted into an FHE-compatible quantized module using the [compile_brevitas_qat_model](https://github.com/aaravm/watermark/blob/master/concrete-model.py#L54) function. The parameters used here are: <br>
        a. show_mlir=False,<br>
        b. output_onnx_file="test.onnx",<br>
        c. rounding_threshold_bits={"n_bits": 8, "method": Exactness.APPROXIMATE},<br>
        d. configuration=config,<br>
        e. verbose=True,<br>
        f. device="cpu"<br>
3. **Performing inference:** Using FHE simulation since it is faster than the actual FHE compilation, because it relies only on Python, it doesn't perform any crypto operations or keys generation, on the qmodel generated at [Line 201](https://github.com/aaravm/watermark/blob/master/concrete-model.py#L201)


## Performance
The `test_results` directory contains the original and watermarked images for the first batch of the test-dataset (1000 32x32 images). Performance metrics, including bitwise accuracy across different epochs for all models, will be provided to evaluate the system's effectiveness.

We also report the performances of our models in terms of BER (Bit error rate) and PSNR (Peak Signal to Noise Ratio) on a set of 1000 32x32 images.


|  | "QuantWaveTF32x32" | "QuantizedMLPModel" | 
|----------------|----------------|----------------|
|PSNR  |31.09        | 18.49         |
| BER  |  10.48        | 50.50         | 

### Metrics:
![image](https://github.com/user-attachments/assets/1a5da145-590d-46b0-8415-d13079d6846e)

### Illustration of Invisibility of Watermarked Images:
![image](https://github.com/user-attachments/assets/467790bd-dd09-4ad9-9788-da8c86be9684) ------> ![image](https://github.com/user-attachments/assets/c4d073ad-16e5-4205-8da0-0d56a7949a9a)

                                                                                          









## Understanding the structure of the repository:
1. `models` folder contains both the models, `QuantizedMLPModel` and `QuantWaveTF32x32` along with their pyTorch equivalents

2. `data_loaders` Combines images and their corresponding random watermarks into a unified dataloader.

3. `concrete-model.py` The main model script where training is conducted using QAT. It also performs FHE simulation for inference without requiring key generation.

4. `test_results` Contains the original and watermarked images for the first batch.
5. `utils.py` contains the functions of metric used to evaluate the performance of our models such as PSNR (Peak Signal to Noise Ratio) and BER (Bit Error Rate)

## Training data
You can find the link to directly download the data [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).
Also to find out more about the dataset and its specification, have a read [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

## Getting Started
### Prerequisites
<li>Python 3.8 or higher
<li>PyTorch
<li>Concrete ML

## Installation

### Step 1: Clone the repository:
```
git clone https://github.com/aaravm/watermark.git
cd watermark
```

### Step 2: Install Concrete-ML
Install the `concrete-ml` package using `pip`:
```bash
pip install concrete-ml
```

### Step 3: Install PyTorch Wavelets
Clone the `pytorch_wavelets` repository and install it:
```bash
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```

### Step 4: Install PyWavelets
Install the `PyWavelets` package using `pip`:
```bash
pip install PyWavelets
```

### Step 5: Install PyTorch and Related Packages
Install `PyTorch`, `torchvision`, and `torchaudio` using `conda`:
```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 cpuonly -c pytorch
```

### Step 6: Download the dataset
Download the dataset and save it in a `train` folder in the main repository

### Step 7: Run the Training Script
Run the training script to train the model:
```bash
python concrete-model.py
```
Note: This python model is trained with a cpu, not cuda


