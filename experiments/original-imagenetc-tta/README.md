Minimal ImageNet-C TTA Example
================================

Purpose
-------
This folder provides the loose-file ImageNet-C examples:

- `minimal_imagenetc_tta_example.py`: the original minimal baseline.
- `minimal_imagenetc_tta_metrics_example.py`: logs inode and throughput metrics
  for the loose-file dataset layout.

For the archive-backed variant, see:

- `../archive-imagenetc-tta/minimal_imagenetc_tta_archive_metrics_example.py`
- `../archive-imagenetc-tta/README.md`

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

Execution
---------
1. Install requirements:

   ```bash
   pip install torch torchvision
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

What the loose-file metrics script logs
---------------------------------------
`minimal_imagenetc_tta_metrics_example.py` reports:

- Dataset size/index stats
- Inode-object approximation and reduction factor
- Data pipeline setup/indexing timings
- End-to-end forward pass wall time and throughput (`images/s`)
- Time-to-first-batch and first-batch compute time
- Peak CUDA memory (when running on GPU)
