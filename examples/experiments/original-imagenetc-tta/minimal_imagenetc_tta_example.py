import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

"""
Minimal runnable example:
Demonstrates how ImageNet-C is accessed in our TTA evaluation.

This script:
- Loads one corruption type and severity level
- Performs forward passes only
- Does NOT create or write any files
"""

# -------- CONFIGURATION --------
# Adjust this path to your ImageNet-C root directory
IMAGENET_C_ROOT = "/path/to/imagenet-c"

CORRUPTION = "fog"
SEVERITY = "5"

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------


def main():

    data_path = os.path.join(IMAGENET_C_ROOT, CORRUPTION, SEVERITY)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # Example model (ResNet-18)
    model = torchvision.models.resnet18(pretrained=True)
    model.to(DEVICE)
    model.eval()

    print(f"Dataset size: {len(dataset)} images")
    print("Starting forward pass...")

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)

    print("Finished evaluation.")
    print("No files were written during execution.")


if __name__ == "__main__":
    main()
