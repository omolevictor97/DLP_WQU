import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mean_and_std(dataloader):
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0

    for images, _ in dataloader:
        # Assuming images are in (Batch, Channel, Height, Width) format
        # Calculate sum and sum of squares for each channel
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.sqrt(channels_squared_sum / num_batches - mean**2)

    return mean, std

# Example usage with a dummy dataset
# Define a transform to convert images to tensors
transform = transforms.ToTensor()

# Create a dummy dataset (replace with your actual dataset)
# For demonstration, we'll use a random tensor for a single batch
dummy_dataset = [torch.rand(3, 64, 64) for _ in range(100)] # 100 images, 3 channels, 64x64
dummy_dataloader = DataLoader(dummy_dataset, batch_size=32)

# Calculate mean and std
dataset_mean, dataset_std = get_mean_and_std(dummy_dataloader)

print(f"Dataset Mean: {dataset_mean}")
print(f"Dataset Standard Deviation: {dataset_std}")