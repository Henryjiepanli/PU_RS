# PUGNet

This repository provides the official implementation of our paper:

**Progressive Uncertainty-Guided Network for Binary Segmentation in High-Resolution Remote Sensing Imagery**

PUGNet is designed to explicitly model and exploit predictive uncertainty for accurate binary segmentation in high-resolution remote sensing images. The framework supports both **single-temporal semantic segmentation** and **bi-temporal change detection** scenarios.

---

## 📁 Repository Structure

The codebase is organized into two main modules according to the task setting:

1. **Single-Temporal Binary Segmentation**  
   Binary segmentation (e.g., buildings, cropland) from single-temporal optical imagery.  
   📂 `single-temporal_binary_segmentation/`

2. **Bi-Temporal Binary Segmentation**  
   Binary change detection (e.g., building change detection) from bi-temporal optical imagery.  
   📂 `bi-temporal_binary_segmentation/`

---

## 🚀 Single-Temporal Binary Segmentation

### Training

Navigate to the `single-temporal_binary_segmentation` directory and run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c config/whubuilding/PUGNet.py
CUDA_VISIBLE_DEVICES=0 python train.py -c config/massbuilding/PUGNet.py
CUDA_VISIBLE_DEVICES=0 python train.py -c config/inriabuilding/PUGNet.py
CUDA_VISIBLE_DEVICES=0 python train.py -c config/fgfd/PUGNet.py
