Minimal ImageNet-C TTA Example
================================

Purpose
-------
This folder provides two scripts:

- `minimal_imagenetc_tta_example.py`: the original minimal baseline.
- `minimal_imagenetc_tta_metrics_example.py`: a second version that adds
  inode and throughput-oriented logging for archive-migration planning.

Dataset Structure
------------------
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

What the metrics script logs
----------------------------
`minimal_imagenetc_tta_metrics_example.py` reports metrics that are useful
when evaluating a switch from many loose image files to archive-based datasets:

- Indexed image count and class count
- Estimated dataset volume (GiB)
- Directory count and approximate inode objects (`files + dirs`)
- Projected inode reduction factor for a configurable number of archive shards
- Data pipeline setup/indexing timings
- End-to-end forward pass time and throughput (`images/s`)
- Time-to-first-batch and first-batch compute time
- Peak CUDA memory (when running on GPU)

Execution
---------
1. Install requirements:

   pip install torch torchvision

2. Adjust the dataset path in the script you want to run:

   IMAGENET_C_ROOT = "/path/to/imagenet-c"

3. Run the original baseline:

   python minimal_imagenetc_tta_example.py

4. Run the metrics-enriched variant:

   python minimal_imagenetc_tta_metrics_example.py

5. Optional (metrics script only): set expected archive shard count:

   EXPECTED_ARCHIVE_SHARDS = 16
