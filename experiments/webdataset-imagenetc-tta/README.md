WebDataset-Streaming ImageNet-C TTA Example
===========================================

Purpose
-------
This folder provides an additional experiment that uses **WebDataset-style tar
streaming** for ImageNet-C/Tiny-ImageNet-C shards:

- `minimal_imagenetc_tta_webdataset_metrics_example.py`

The script keeps the same forward-pass-only observability style as the other
experiments, but switches the input pipeline to a DDP-friendly iterable stream
with:

- shard shuffle (`shardshuffle`)
- sample buffer shuffle (`shuffle`)
- multi-worker streaming (`WebLoader`)

Why this experiment is useful for inode pressure
------------------------------------------------
- You still store data in a manageable number of tar shards (thousands instead
  of millions of files), so inode pressure is dramatically lower.
- Reads are mostly sequential at the tar level, which generally improves
  streaming throughput.
- The pipeline is widely used for large-scale training.

- The loader discovers existing `*.tar` files directly and does not assume a contiguous
  shard index range (so missing shard numbers are fine).

Main tradeoff
-------------
- Sampling is **approximate shuffle**, not perfect global random sampling.
  In practice this is usually acceptable for vision training.

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

Inside each shard, image paths should preserve class directories, e.g.
`n01440764/some_image.JPEG`.

Create tar shards (if needed)
-----------------------------
Reuse the existing conversion utility from the archive experiment folder.

### Convert one split (`fog/5`)

```bash
python ../archive-imagenetc-tta/convert_imagenetc_split_to_tar_shards.py \
  --input-root /dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C/fog/5 \
  --output-root /dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C-archives/fog/5 \
  --num-shards 16
```

### Convert full Tiny-ImageNet-C (all corruption/severity groups)

```bash
python ../archive-imagenetc-tta/convert_imagenetc_split_to_tar_shards.py \
  --input-root /dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C \
  --output-root /dss/dsshome1/05/di38qex/datasets/Tiny-ImageNet/Tiny-ImageNet-C-archives \
  --group-depth 2 \
  --num-shards 16
```

Execution
---------
1. Install requirements:

   ```bash
   pip install torch torchvision webdataset pillow
   ```

2. In `minimal_imagenetc_tta_webdataset_metrics_example.py`, set:

   - `IMAGENET_C_ARCHIVE_ROOT`
   - `CORRUPTION`
   - `SEVERITY`

3. Run the experiment:

   ```bash
   python minimal_imagenetc_tta_webdataset_metrics_example.py
   ```

What the script logs
--------------------
- Dataset size/index stats
- Inode-object approximation and reduction factor
- Metadata scan and datapipe setup timings
- End-to-end forward pass wall time and throughput (`images/s`)
- Time-to-first-batch and first-batch compute time
- Peak CUDA memory (when running on GPU)
- Streaming shuffle configuration (shard/buffer shuffle)
