# torch-archive-dataloader

Minimal ImageNet-C TTA Example
================================

Purpose
-------
This repository now includes a baseline ImageNet-C example (individual files)
plus additional examples to explore alternative storage and loading patterns.

The original baseline script is kept as-is and can still be run directly.

Dataset Structure (baseline)
----------------------------
ImageNet-C follows the standard corruption structure:

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

Each corruption type and severity level is treated as a separate
ImageFolder dataset.

Examples
--------

### 1) Baseline: Individual files (original script, unchanged)

File:

```text
examples/minimal_imagenetc_tta_example.py
```

Run directly:

```bash
python examples/minimal_imagenetc_tta_example.py
```

### 2) Alternative: Tar archive-backed dataset

File:

```text
examples/imagenetc_tta_tar_archive_example.py
```

This demonstrates loading images from a `.tar` archive that preserves the
class-folder layout.

Run directly:

```bash
python examples/imagenetc_tta_tar_archive_example.py
```

### 3) Alternative: Eager in-memory cache

File:

```text
examples/imagenetc_tta_memory_cache_example.py
```

This demonstrates front-loading file I/O by loading images into memory during
startup, then iterating in-memory.

Run directly:

```bash
python examples/imagenetc_tta_memory_cache_example.py
```

### Unified launcher (mode selector)

You can also choose an approach through one entrypoint:

```bash
python -m examples.run_imagenetc_example --mode individual-files
python -m examples.run_imagenetc_example --mode tar-archive
python -m examples.run_imagenetc_example --mode memory-cache
```

Execution
---------
1. Install requirements:

   ```bash
   pip install torch torchvision pillow
   ```

2. Adjust dataset paths in each selected example script, e.g.:

   ```python
   IMAGENET_C_ROOT = "/path/to/imagenet-c"
   ```

3. Run your selected mode directly or via the unified launcher.
