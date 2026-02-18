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
Archive-based metrics example for full ImageNet-C/Tiny-ImageNet-C evaluation.

This script runs forward-pass-only evaluation over every corruption/severity split
found under an archive root laid out as:

    IMAGENET_C_ARCHIVE_ROOT/
        <corruption>/
            <severity>/
                shard-00000.tar
                ...
"""

# -------- CONFIGURATION --------
IMAGENET_C_ARCHIVE_ROOT = "/dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C-archives/"

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
                    self.samples.append((archive_idx, member.name, 0, member.size))

        self.classes = sorted(class_names)
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

        indexed_samples = []
        for archive_idx, member_name, _, member_size in self.samples:
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


def _discover_splits(archive_root: Path) -> List[Tuple[str, str, Path]]:
    splits: List[Tuple[str, str, Path]] = []
    for corruption_dir in sorted(path for path in archive_root.iterdir() if path.is_dir()):
        for severity_dir in sorted(path for path in corruption_dir.iterdir() if path.is_dir()):
            if any(severity_dir.glob("*.tar")):
                splits.append((corruption_dir.name, severity_dir.name, severity_dir))
    return splits


def _run_forward_pass(loader: DataLoader, model: torch.nn.Module) -> Tuple[int, int, float]:
    images_processed = 0
    total_batches = 0

    pass_start = time.perf_counter()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            _ = model(images)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            images_processed += images.size(0)
            total_batches += 1
    pass_total_s = time.perf_counter() - pass_start

    return images_processed, total_batches, pass_total_s


def main() -> None:
    archive_root = Path(IMAGENET_C_ARCHIVE_ROOT)
    if not archive_root.exists():
        raise FileNotFoundError(f"Archive root not found: {archive_root}")

    splits = _discover_splits(archive_root)
    if not splits:
        raise RuntimeError(f"No archive splits found under {archive_root}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(DEVICE)
    model.eval()

    print("=== Full Dataset Archive Evaluation ===")
    print(f"Archive root: {archive_root}")
    print(f"Detected splits: {len(splits)}")
    print(f"Device: {DEVICE}")
    print()

    total_images = 0
    total_batches = 0
    total_pass_s = 0.0

    for corruption, severity, split_path in splits:
        t0 = time.perf_counter()
        dataset = TarImageFolderDataset(archive_dir=split_path, transform=transform)
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

        images_processed, split_batches, split_pass_s = _run_forward_pass(loader=loader, model=model)
        split_throughput = images_processed / split_pass_s if split_pass_s > 0 else float("nan")

        total_images += images_processed
        total_batches += split_batches
        total_pass_s += split_pass_s

        print(f"[{corruption}/{severity}]")
        print(f"  Classes: {len(dataset.classes)}")
        print(f"  Archive shards: {len(dataset.archive_paths)}")
        print(f"  Images: {images_processed}")
        print(f"  Dataset init: {dataset_init_s:.3f}s")
        print(f"  DataLoader init: {dataloader_init_s:.3f}s")
        print(f"  Forward pass: {split_pass_s:.3f}s")
        print(f"  Throughput: {split_throughput:.2f} images/s")
        print()

    global_throughput = total_images / total_pass_s if total_pass_s > 0 else float("nan")

    print("=== Aggregate Summary ===")
    print(f"Splits evaluated: {len(splits)}")
    print(f"Total batches: {total_batches}")
    print(f"Total images processed: {total_images}")
    print(f"Total forward-pass wall time: {total_pass_s:.3f}s")
    print(f"Overall throughput: {global_throughput:.2f} images/s")


if __name__ == "__main__":
    main()
