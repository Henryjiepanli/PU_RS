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
```

### Testing

Navigate to the `single-temporal_binary_segmentation` directory and run:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py -c config/whubuilding/PUGNet.py -o test_results/whubuilding/PUGNet/ -m model_path --rgb
CUDA_VISIBLE_DEVICES=0 python test.py -c config/massbuilding/PUGNet.py -o test_results/massbuilding/PUGNet/ -m model_path --rgb
CUDA_VISIBLE_DEVICES=0 python test.py -c config/inriabuilding/PUGNet.py -o test_results/inriabuilding/PUGNet/ -m model_path --rgb
CUDA_VISIBLE_DEVICES=0 python test.py -c config/fgfd/PUGNet.py -o test_results/fgfd/PUGNet/ -m model_path --rgb
```

### Test Time Augmentation (TTA)

To enable Test Time Augmentation (TTA) during inference, use the `-t` option:

```bash
python test.py -c config/whubuilding/PUGNet.py \
               -o test_results/whubuilding/PUGNet/ \
               -m model_path \
               -t lr \
               --rgb
```

### Continuing Training / Multiple Training Sessions
To resume training from a checkpoint or conduct multiple training sessions:

1.Specify the checkpoint path in the configuration file:

```yaml
pretrained_ckpt_path: path/to/checkpoint.pth
```

2.Re-run the training script as usual.

## 🚀 Single-Temporal Binary Segmentation

### Training
Navigate to the `bi-temporal_binary_segmentation` directory and run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --data_name LEVIR-CD
CUDA_VISIBLE_DEVICES=0 python train.py --data_name Google
CUDA_VISIBLE_DEVICES=0 python train.py --data_name SYSU
CUDA_VISIBLE_DEVICES=0 python train.py --data_name WHU
CUDA_VISIBLE_DEVICES=0 python train.py --data_name Lebedev
CUDA_VISIBLE_DEVICES=0 python train.py --data_name LEVIR-CD+
```

### Testing

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --data_name LEVIR-CD
CUDA_VISIBLE_DEVICES=0 python test.py --data_name Google
CUDA_VISIBLE_DEVICES=0 python test.py --data_name SYSU
CUDA_VISIBLE_DEVICES=0 python test.py --data_name WHU
CUDA_VISIBLE_DEVICES=0 python test.py --data_name Lebedev
CUDA_VISIBLE_DEVICES=0 python test.py --data_name LEVIR-CD+
```

## 📦 Pretrained Backbones
We provide the pretrained `PVT-v2` backbone used in our experiments.

- [Download Link](https://pan.baidu.com/s/1FdtPvY1TRd8UiL-_Yf9cdQ?pwd=dadp)  
  Code: `dadp `

## 📊 Test Results

We also release the prediction results reported in the paper:

**Building Segmentation**
- [Download Link]( https://pan.baidu.com/s/1IJ0DfinPDUKGPowbD0q8gg?pwd=dadp)  
  Code: `dadp`
  
**Cropland Segmentation**
- [Download Link](https://pan.baidu.com/s/1QvqOX4YSgZ5HtWuh2pxgdQ?pwd=dadp)  
  Code: `dadp`
  
**Building Change Detection**
- [Download Link](https://pan.baidu.com/s/1yNC4kWF4cPbHHFXPUpOrdw?pwd=dadp)  
  Code: `dadp`

## ⚠️ Notes

- Ensure all required dependencies and environment configurations are correctly installed before running the code.

- The configuration files fully specify dataset paths, training schedules, and uncertainty-related settings.
