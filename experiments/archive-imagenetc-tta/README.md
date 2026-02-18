Archive-backed ImageNet-C TTA Example
=====================================

Purpose
-------
This folder provides an archive-backed (tar shard) ImageNet-C experiment:

- `minimal_imagenetc_tta_archive_metrics_example.py`

The script keeps the same observability style as the loose-file metrics
experiment while loading images from `.tar` shards through a custom PyTorch
`Dataset` and `DataLoader`.

Expected Archive Structure
--------------------------
The script expects this layout:

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
Use the dedicated conversion script in this folder:

```bash
python convert_imagenetc_split_to_tar_shards.py \
  --input-root /dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C/fog/5 \
  --output-root /dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C-archives/fog/5 \
  --num-shards 16
```

This converts one corruption/severity split from loose files to `.tar` shards
while preserving class-relative paths.

Execution
---------
1. Install requirements:

   ```bash
   pip install torch torchvision pillow
   ```

2. Convert the loose-file split to archive shards:

   ```bash
   python convert_imagenetc_split_to_tar_shards.py \
     --input-root /dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C/fog/5 \
     --output-root /dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C-archives/fog/5 \
     --num-shards 16
   ```

3. In the metrics script, use archive dataset paths under:

   ```text
   /dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C-archives
   ```

4. Run:

   ```bash
   python minimal_imagenetc_tta_archive_metrics_example.py
   ```

What the script logs
--------------------
- Dataset size/index stats
- Inode-object approximation and reduction factor
- Data pipeline setup/indexing timings
- End-to-end forward pass wall time and throughput (`images/s`)
- Time-to-first-batch and first-batch compute time
- Peak CUDA memory (when running on GPU)
