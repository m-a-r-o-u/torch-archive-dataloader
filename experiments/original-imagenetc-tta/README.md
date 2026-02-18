Minimal ImageNet-C TTA Example
================================

Purpose
-------
This folder provides three scripts:

- `minimal_imagenetc_tta_example.py`: the original minimal baseline.
- `minimal_imagenetc_tta_metrics_example.py`: logs inode and throughput metrics
  for the loose-file dataset layout.
- `minimal_imagenetc_tta_archive_metrics_example.py`: logs the same style of
  metrics while loading from `.tar` archive shards.

Dataset Structure
------------------
ImageNet-C loose-file layout:

```text
imagenet-c/
    fog/
        1/
            n01440764/
                *.JPEG
        2/
        ...
    gaussian_noise/
    motion_blur/
    ...
```

Archive layout expected by `minimal_imagenetc_tta_archive_metrics_example.py`:

```text
imagenet-c-archives/
    fog/
        5/
            shard-00000.tar
            shard-00001.tar
            ...
```

Inside each tar shard, files should preserve relative class paths, e.g.
`n01440764/some_image.JPEG`.

How to convert loose files to tar shards
----------------------------------------
The command below converts one corruption/severity split into archive shards.
It keeps class subpaths and creates roughly even shards.

```bash
python - <<'PY'
import math
import tarfile
from pathlib import Path

INPUT_ROOT = Path('/path/to/imagenet-c/fog/5')
OUTPUT_ROOT = Path('/path/to/imagenet-c-archives/fog/5')
NUM_SHARDS = 16

image_paths = sorted([
    p for p in INPUT_ROOT.rglob('*')
    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
])

if not image_paths:
    raise RuntimeError(f'No images found in {INPUT_ROOT}')

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
shard_size = math.ceil(len(image_paths) / NUM_SHARDS)

for shard_idx in range(NUM_SHARDS):
    start = shard_idx * shard_size
    end = min(start + shard_size, len(image_paths))
    if start >= len(image_paths):
        break

    shard_path = OUTPUT_ROOT / f'shard-{shard_idx:05d}.tar'
    with tarfile.open(shard_path, 'w') as tar:
        for img_path in image_paths[start:end]:
            arcname = img_path.relative_to(INPUT_ROOT)
            tar.add(img_path, arcname=str(arcname))

print(f'Wrote shards to: {OUTPUT_ROOT}')
PY
```

Execution
---------
1. Install requirements:

   ```bash
   pip install torch torchvision pillow
   ```

2. Adjust dataset path(s) in the script you want to run.

3. Run the original baseline:

   ```bash
   python minimal_imagenetc_tta_example.py
   ```

4. Run the loose-file metrics variant:

   ```bash
   python minimal_imagenetc_tta_metrics_example.py
   ```

5. Run the archive metrics variant:

   ```bash
   python minimal_imagenetc_tta_archive_metrics_example.py
   ```

What the metrics scripts log
----------------------------
Both metrics scripts report the same categories of signals:

- Dataset size/index stats
- Inode-object approximation and reduction factor
- Data pipeline setup/indexing timings
- End-to-end forward pass wall time and throughput (`images/s`)
- Time-to-first-batch and first-batch compute time
- Peak CUDA memory (when running on GPU)
