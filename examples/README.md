Minimal ImageNet-C TTA Example
================================

Purpose
-------
This example demonstrates how the ImageNet-C dataset is accessed
during our test-time adaptation (TTA) experiments.

Dataset Structure
------------------
ImageNet-C follows the standard corruption structure:

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

Each corruption type and severity level is treated as a separate
ImageFolder dataset.

Execution
---------
1. Install requirements:

   pip install torch torchvision

2. Adjust the dataset path in:

   IMAGENET_C_ROOT = "/path/to/imagenet-c"

3. Run:

   python minimal_imagenetc_tta_example.py
