import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
import webdataset as wds
from torchvision.models import ResNet18_Weights

"""
WebDataset-style tar-streaming metrics example for ImageNet-C TTA.

This script demonstrates a battle-tested streaming pipeline:
- shard-level shuffle (`shardshuffle`)
- buffer shuffle (`shuffle`)
- worker splitting (`split_by_worker`)
- decode + transform + batch

It keeps the same forward-pass-only observability style as the other experiments
while using iterable tar streaming instead of a custom map-style Dataset.
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

# Approximate-shuffling controls (common WebDataset setup).
SHARD_SHUFFLE = 64
SAMPLE_SHUFFLE = 1000
# --------------------------------

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    return f"{gib:.2f} GiB"


def _scan_archive_metadata(archive_dir: Path) -> Tuple[List[str], int, int, int, int]:
    import tarfile

    archive_paths = sorted(archive_dir.glob("*.tar"))
    if not archive_paths:
        raise FileNotFoundError(f"No .tar shards found in {archive_dir}")

    class_names = set()
    image_count = 0
    estimated_member_bytes = 0
    total_archive_bytes = 0

    for tar_path in archive_paths:
        total_archive_bytes += tar_path.stat().st_size
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                suffix = Path(member.name).suffix.lower()
                if suffix not in VALID_EXTENSIONS:
                    continue

                class_name = Path(member.name).parent.name
                class_names.add(class_name)
                image_count += 1
                estimated_member_bytes += member.size

    return (
        sorted(class_names),
        image_count,
        estimated_member_bytes,
        total_archive_bytes,
        len(archive_paths),
    )


def _extract_class_key(sample: Dict[str, object]) -> str:
    sample_key = sample["__key__"]
    if not isinstance(sample_key, str):
        raise ValueError("Expected string __key__ in WebDataset sample")

    key_parts = Path(sample_key).parts
    if len(key_parts) < 2:
        raise ValueError(
            f"Unable to infer class from key '{sample_key}'. "
            "Expected keys like 'n01440764/image_0001'."
        )
    return key_parts[-2]


def main() -> None:
    archive_path = Path(IMAGENET_C_ARCHIVE_ROOT) / CORRUPTION / SEVERITY
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive path not found: {archive_path}")

    t0 = time.perf_counter()
    (
        class_names,
        expected_images,
        estimated_member_bytes,
        total_archive_bytes,
        archive_file_count,
    ) = _scan_archive_metadata(archive_path)
    metadata_scan_s = time.perf_counter() - t0

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    inode_objects_on_disk = archive_file_count + 1
    projected_loose_objects = expected_images + len(class_names) + 1
    inode_reduction_factor = projected_loose_objects / inode_objects_on_disk

    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    shard_urls = [str(path) for path in sorted(archive_path.glob("*.tar"))]


    def convert_sample(sample: Dict[str, object]):
        class_name = _extract_class_key(sample)
        if class_name not in class_to_idx:
            raise KeyError(f"Class '{class_name}' inferred from sample key is not in scanned class list")

        image = sample["jpg"]
        image_tensor = image_transform(image)
        target = class_to_idx[class_name]
        return image_tensor, target

    t1 = time.perf_counter()
    dataset = (
        wds.WebDataset(
            shard_urls,
            shardshuffle=SHARD_SHUFFLE,
            handler=wds.handlers.reraise_exception,
        )
        .shuffle(SAMPLE_SHUFFLE)
        .decode("pil")
        .rename(jpg="jpg;jpeg;png;bmp;webp")
        .map(convert_sample)
        .batched(BATCH_SIZE, partial=False)
    )

    loader = wds.WebLoader(
        dataset,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        batch_size=None,
    )
    datapipe_init_s = time.perf_counter() - t1

    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(DEVICE)
    model.eval()

    print("=== Dataset Access Summary (WebDataset) ===")
    print(f"Path: {archive_path}")
    print(f"Corruption: {CORRUPTION}, severity: {SEVERITY}")
    print(f"Classes: {len(class_names)}")
    print(f"Archive shards: {archive_file_count}")
    print(f"Images indexed from archives: {expected_images}")
    print(f"Estimated raw image payload: {_format_bytes(estimated_member_bytes)}")
    print(f"Archive bytes on disk: {_format_bytes(total_archive_bytes)}")
    print(f"Approx inode objects touched (archives + dir): {inode_objects_on_disk}")
    print(f"Projected loose-file inode objects (images + classes + dir): {projected_loose_objects}")
    print(f"Projected inode reduction factor vs loose files: {inode_reduction_factor:.1f}x")
    print()

    print("=== Initialization Timings ===")
    print(f"Archive metadata scan time: {metadata_scan_s:.3f}s")
    print(f"WebDataset/WebLoader setup time: {datapipe_init_s:.3f}s")
    print()

    print("=== Streaming Configuration ===")
    print(f"Shard shuffle buffer: {SHARD_SHUFFLE}")
    print(f"Sample shuffle buffer: {SAMPLE_SHUFFLE}")
    print("Sampling: approximate shuffle (shard + buffer), DDP-friendly iterable stream")
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
    print(f"Expected images from metadata scan: {expected_images}")
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
