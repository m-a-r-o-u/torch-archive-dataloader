import os
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights

"""
Metrics-enriched runnable example:
Demonstrates how ImageNet-C is accessed in our TTA evaluation while logging
signals that are useful when planning a migration from loose files to archive
shards to mitigate inode pressure.

This script:
- Loads one corruption type and severity level
- Performs forward passes only
- Does NOT create or write any files
- Logs dataset/inode and throughput metrics
"""

# -------- CONFIGURATION --------
# Adjust this path to your ImageNet-C root directory
IMAGENET_C_ROOT = "/dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C/"

CORRUPTION = "fog"
SEVERITY = "5"

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
# If you are planning archive shards, set this to your expected shard count.
# Example: 10_000 loose files -> 16 tar shards reduces inode objects by ~625x.
EXPECTED_ARCHIVE_SHARDS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------


def _count_directories(root: Path) -> int:
    return sum(1 for entry in root.rglob("*") if entry.is_dir())


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    return f"{gib:.2f} GiB"


def main() -> None:
    data_path = Path(IMAGENET_C_ROOT) / CORRUPTION / SEVERITY
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    t0 = time.perf_counter()
    dataset = ImageFolder(root=str(data_path), transform=transform)
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

    sample_file_count = len(dataset.samples)
    class_count = len(dataset.classes)
    directory_count = _count_directories(data_path)
    inode_objects_on_disk = sample_file_count + directory_count
    projected_archive_objects = EXPECTED_ARCHIVE_SHARDS + 1  # + root directory
    inode_reduction_factor = inode_objects_on_disk / projected_archive_objects

    # Estimation based on file sizes that ImageFolder already indexed.
    total_dataset_bytes = sum(os.path.getsize(path) for path, _ in dataset.samples)

    # Example model (ResNet-18)
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(DEVICE)
    model.eval()

    print("=== Dataset Access Summary ===")
    print(f"Path: {data_path}")
    print(f"Corruption: {CORRUPTION}, severity: {SEVERITY}")
    print(f"Classes: {class_count}")
    print(f"Image files indexed: {sample_file_count}")
    print(f"Estimated dataset volume: {_format_bytes(total_dataset_bytes)}")
    print(f"Directory count under split: {directory_count}")
    print(f"Approx inode objects touched (files + dirs): {inode_objects_on_disk}")
    print(f"Projected archive objects ({EXPECTED_ARCHIVE_SHARDS} shards + root): {projected_archive_objects}")
    print(f"Projected inode reduction factor: {inode_reduction_factor:.1f}x")
    print()

    print("=== Initialization Timings ===")
    print(f"ImageFolder indexing time: {dataset_init_s:.3f}s")
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
