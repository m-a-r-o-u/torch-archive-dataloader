import os
from typing import List, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

"""
Alternative example:
Demonstrates eager in-memory caching of one corruption/severity split.

This is still backed by the file layout used by ImageFolder, but loading
is front-loaded during dataset construction to reduce repeated file I/O.
"""

# -------- CONFIGURATION --------
IMAGENET_C_ROOT = "/path/to/imagenet-c"
CORRUPTION = "fog"
SEVERITY = "5"

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------


class MemoryCachedImageFolder(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: List[Tuple[Image.Image, int]] = []

        class_names = sorted(
            [entry.name for entry in os.scandir(root) if entry.is_dir()]
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(root, class_name)
            label = self.class_to_idx[class_name]

            for file_entry in os.scandir(class_dir):
                if not file_entry.is_file():
                    continue
                if not file_entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image = Image.open(file_entry.path).convert("RGB")
                self.samples.append((image.copy(), label))
                image.close()

        if not self.samples:
            raise RuntimeError(f"No images found in: {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def main():
    data_path = os.path.join(IMAGENET_C_ROOT, CORRUPTION, SEVERITY)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = MemoryCachedImageFolder(root=data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

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
