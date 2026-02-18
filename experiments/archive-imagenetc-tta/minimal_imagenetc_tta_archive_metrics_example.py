import io
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights

"""
Archive-based metrics example for ImageNet-C TTA.

This script demonstrates how to run the same forward-pass evaluation flow when
the dataset is stored as tar archives instead of loose image files.

This script:
- Loads one corruption type and severity level from tar shards
- Performs forward passes only
- Does NOT create or write any files
- Logs dataset/inode and throughput metrics
"""

# -------- CONFIGURATION --------
# Expected archive layout:
#   IMAGENET_C_ARCHIVE_ROOT/
#       fog/
#           5/
#               shard-00000.tar
#               shard-00001.tar
#               ...
IMAGENET_C_ARCHIVE_ROOT = "/dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C-archives/"

CORRUPTION = "fog"
SEVERITY = "5"

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TarImageFolderDataset(Dataset):
    def __init__(self, archive_dir: Path, transform=None) -> None:
        self.archive_dir = archive_dir
        self.transform = transform
        self.archive_paths = sorted(self.archive_dir.glob("*.tar"))
        if not self.archive_paths:
            raise FileNotFoundError(f"No .tar shards found in {self.archive_dir}")

        self.samples: List[Tuple[int, str, int, int]] = []
        class_names = set()
        self.total_archive_bytes = 0

        for archive_idx, tar_path in enumerate(self.archive_paths):
            self.total_archive_bytes += tar_path.stat().st_size
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    suffix = Path(member.name).suffix.lower()
                    if suffix not in VALID_EXTENSIONS:
                        continue

                    class_name = Path(member.name).parent.name
                    class_names.add(class_name)
                    self.samples.append((archive_idx, member.name, member.size))

        self.classes = sorted(class_names)
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

        # Replace placeholder targets with resolved class indices.
        indexed_samples = []
        for archive_idx, member_name, member_size in self.samples:
            class_name = Path(member_name).parent.name
            target = self.class_to_idx[class_name]
            indexed_samples.append((archive_idx, member_name, target, member_size))
        self.samples = indexed_samples

        self._tar_by_worker: Dict[Tuple[int, int], tarfile.TarFile] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_tar(self, archive_idx: int) -> tarfile.TarFile:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        cache_key = (worker_id, archive_idx)

        if cache_key not in self._tar_by_worker:
            self._tar_by_worker[cache_key] = tarfile.open(self.archive_paths[archive_idx], "r")
        return self._tar_by_worker[cache_key]

    def __getitem__(self, index: int):
        archive_idx, member_name, target, _ = self.samples[index]
        tar = self._get_tar(archive_idx)

        file_obj = tar.extractfile(member_name)
        if file_obj is None:
            raise FileNotFoundError(f"Missing member {member_name} in {self.archive_paths[archive_idx]}")

        image_bytes = file_obj.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    return f"{gib:.2f} GiB"


def main() -> None:
    archive_path = Path(IMAGENET_C_ARCHIVE_ROOT) / CORRUPTION / SEVERITY
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive path not found: {archive_path}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    t0 = time.perf_counter()
    dataset = TarImageFolderDataset(archive_dir=archive_path, transform=transform)
    dataset_init_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    dataloader_init_s = time.perf_counter() - t1

    image_count = len(dataset)
    class_count = len(dataset.classes)
    archive_file_count = len(dataset.archive_paths)
    inode_objects_on_disk = archive_file_count + 1  # archives + split directory

    # This approximates what the loose-file layout would require inodes for.
    # We assume one file per image plus one class directory per class.
    projected_loose_objects = image_count + class_count + 1
    inode_reduction_factor = projected_loose_objects / inode_objects_on_disk

    # Stored member sizes are raw file sizes in the tar stream.
    # Archive file size is usually a bit larger due to tar headers/padding.
    estimated_member_bytes = sum(member_size for _, _, _, member_size in dataset.samples)

    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(DEVICE)
    model.eval()

    print("=== Dataset Access Summary (Archive) ===")
    print(f"Path: {archive_path}")
    print(f"Corruption: {CORRUPTION}, severity: {SEVERITY}")
    print(f"Classes: {class_count}")
    print(f"Archive shards: {archive_file_count}")
    print(f"Images indexed from archives: {image_count}")
    print(f"Estimated raw image payload: {_format_bytes(estimated_member_bytes)}")
    print(f"Archive bytes on disk: {_format_bytes(dataset.total_archive_bytes)}")
    print(f"Approx inode objects touched (archives + dir): {inode_objects_on_disk}")
    print(f"Projected loose-file inode objects (images + classes + dir): {projected_loose_objects}")
    print(f"Projected inode reduction factor vs loose files: {inode_reduction_factor:.1f}x")
    print()

    print("=== Initialization Timings ===")
    print(f"Archive index build time: {dataset_init_s:.3f}s")
    print(f"DataLoader setup time: {dataloader_init_s:.3f}s")
    print()

    print("Starting forward pass...")

    images_processed = 0
    total_batches = 0
    first_batch_fetch_s = None
    first_batch_compute_s = None

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pass_start = time.perf_counter()
    last_batch_done = pass_start
    with torch.no_grad():
        for images, _ in loader:
            fetch_done = time.perf_counter()
            if first_batch_fetch_s is None:
                first_batch_fetch_s = fetch_done - last_batch_done

            compute_start = time.perf_counter()
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            _ = model(images)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            compute_done = time.perf_counter()

            if first_batch_compute_s is None:
                first_batch_compute_s = compute_done - compute_start

            images_processed += images.size(0)
            total_batches += 1
            last_batch_done = compute_done

    pass_total_s = time.perf_counter() - pass_start
    throughput = images_processed / pass_total_s if pass_total_s > 0 else float("nan")

    print("Finished evaluation.")
    print("No files were written during execution.")
    print()

    print("=== Forward Pass Timings ===")
    print(f"Batches: {total_batches}")
    print(f"Images processed: {images_processed}")
    print(f"Total forward-pass wall time: {pass_total_s:.3f}s")
    print(f"End-to-end throughput: {throughput:.2f} images/s")
    if first_batch_fetch_s is not None:
        print(f"Time-to-first-batch (data loading): {first_batch_fetch_s:.3f}s")
    if first_batch_compute_s is not None:
        print(f"First-batch model compute time: {first_batch_compute_s:.3f}s")

    if DEVICE == "cuda":
        peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak CUDA memory allocated: {peak_mem_gib:.2f} GiB")


if __name__ == "__main__":
    main()
