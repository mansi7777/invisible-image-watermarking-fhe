import numpy as np
import math
from sklearn.metrics import mean_squared_error
import torch
import torchvision.utils as vutils
from data_loaders.merged_data_loader import MergedDataLoader
import os
from configs import *


# def calculate_watermark_mismatch(predicted, ground_truth):
#     predicted_binary = (predicted > 0.5).int()  # Convert predictions to binary (threshold at 0.5)
#     ground_truth_binary = (ground_truth > 0.5).int()  # Convert ground truth to binary
#     mismatches = (predicted_binary != ground_truth_binary).sum().item()  # Count mismatched entries
#     return mismatches

def mse_cal(input_img, output_img):
    inp_cnv = np.array(input_img.cpu()).reshape(-1)  # Flatten to 1D
    out_cnv = np.array(output_img.cpu()).reshape(-1)  # Flatten to 1D
    return mean_squared_error(inp_cnv, out_cnv)

def psnr_cal(input_img, output_img, range_num):
    mse_num = mse_cal(input_img, output_img)
    return 10 * math.log10((range_num ** 2) / mse_num) if mse_num > 0 else float('inf')

def ber_cal(input_wm, output_wm):
    input_wm = input_wm.cpu().numpy()  # Convert to NumPy array
    output_wm = output_wm.cpu().numpy()  # Convert to NumPy array
    a = np.sum(np.equal(np.round(output_wm), input_wm))  # Ensure comparison in NumPy
    return 100 - (100 * a / 64)


def torch_inference(model, data_loader, device):
    psnr_list = []
    ber_list = []
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for inference
        for ((input_image, input_watermark, input_attack_id), (output_image, output_watermark)) in data_loader:
            y = model(input_image.to(device), input_watermark.to(device))
            pred_image, pred_watermark = y
            pred_watermark = (pred_watermark > 0.5).int()

            for i in range(input_watermark.shape[0]):
                psnr_list.append(psnr_cal(output_image[i].cpu(), pred_image[i].cpu(), 1))
                ber_list.append(ber_cal(output_watermark[i].cpu(), pred_watermark[i].cpu()))

    return np.mean(psnr_list), np.mean(ber_list)
