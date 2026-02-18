import io
import os
import tarfile
from typing import List, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

"""
Alternative example:
Demonstrates loading one corruption/severity split from a .tar archive
instead of a directory of individual files.

Expected archive layout (same class-folder convention as ImageFolder):
<archive_root>/<CORRUPTION>/<SEVERITY>/<class_name>/*.JPEG
"""

# -------- CONFIGURATION --------
# Path to archive that contains all files for one split.
# Example: /path/to/imagenet-c/fog_5.tar
IMAGENET_C_ARCHIVE = "/path/to/imagenet-c/fog_5.tar"

# Strip this prefix from tar member names before class discovery, if needed.
ARCHIVE_PREFIX = ""

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------


class TarImageDataset(Dataset):
    def __init__(self, archive_path: str, transform=None, archive_prefix: str = ""):
        self.archive_path = archive_path
        self.transform = transform
        self.archive_prefix = archive_prefix.strip("/")

        with tarfile.open(self.archive_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]

        self.samples: List[Tuple[str, int]] = []
        class_names = set()

        for member in members:
            relative = self._normalize_member_name(member.name)
            if relative is None:
                continue
            parts = relative.split("/")
            if len(parts) < 2:
                continue
            class_names.add(parts[0])

        self.classes = sorted(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        for member in members:
            relative = self._normalize_member_name(member.name)
            if relative is None:
                continue
            parts = relative.split("/")
            if len(parts) < 2:
                continue
            class_name = parts[0]
            label = self.class_to_idx[class_name]
            self.samples.append((member.name, label))

        if not self.samples:
            raise RuntimeError(f"No samples found in archive: {self.archive_path}")

    def _normalize_member_name(self, name: str):
        cleaned = name.strip("/")
        if self.archive_prefix:
            prefix = f"{self.archive_prefix}/"
            if not cleaned.startswith(prefix):
                return None
            cleaned = cleaned[len(prefix):]
        return cleaned

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        member_name, label = self.samples[idx]

        with tarfile.open(self.archive_path, "r") as tar:
            extracted = tar.extractfile(member_name)
            if extracted is None:
                raise RuntimeError(f"Could not extract member: {member_name}")
            image_bytes = extracted.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = TarImageDataset(
        archive_path=IMAGENET_C_ARCHIVE,
        transform=transform,
        archive_prefix=ARCHIVE_PREFIX,
    )

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
