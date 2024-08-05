from data_loaders.merged_data_loader import MergedDataLoader
import torch
# Example Configuration
if __name__ == "__main__":
    from data_loaders.configs import SEED, IMAGE_FORMATS
    # Example usage
    image_base_path = "D:/ML/zama/watermark/train_images/abstract_train"
    # image_channels = [0]  # Example channel selection
    image_convert_type = torch.float32
    watermark_size = (8 * 8,)
    attack_min_id = 0
    attack_max_id = 4
    batch_size = 10

    merged_loader = MergedDataLoader(
        image_base_path=image_base_path,
        image_convert_type=image_convert_type,
        watermark_size=watermark_size,
        attack_min_id=attack_min_id,
        attack_max_id=attack_max_id,
        batch_size=batch_size
    )

    data_loader = merged_loader.get_data_loader()

    for batch_idx, ((input_image, input_watermark, input_attack_id), (output_image, output_watermark)) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Input: Image Shape: {input_image.shape}, Watermark Shape: {input_watermark.shape}, Attack ID: {input_attack_id}")
        print(f"Output: Image Shape: {output_image.shape}, Watermark Shape: {output_watermark.shape}")
        if batch_idx >= 2:  # Example: Stop after 3 batches
            break