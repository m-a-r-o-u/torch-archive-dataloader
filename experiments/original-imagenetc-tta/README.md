Minimal ImageNet-C TTA Example
================================

Purpose
-------
This example demonstrates how the ImageNet-C dataset is accessed
during our test-time adaptation (TTA) experiments.

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

What the script logs
--------------------
The example now reports metrics that are useful when evaluating a switch
from many loose image files to archive-based datasets:

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

2. Adjust the dataset path in:

   IMAGENET_C_ROOT = "/path/to/imagenet-c"

3. Optional: set expected archive shard count in the script:

   EXPECTED_ARCHIVE_SHARDS = 16

4. Run:

   python minimal_imagenetc_tta_example.py
