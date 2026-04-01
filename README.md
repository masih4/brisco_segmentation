# 🧠 Breast Tumor Segmentation Pipeline Using an Ensemble of 2.5D Attention U-Net and 3D ResNet Models

> An ensemble segmentation framework combining **2.5D multi-planar** and **3D volumetric** deep learning models for breast tumor segmentation on the ISPY1 DCE-MRI dataset.
>
<p align="center">
  <img src="viz.png" width="900">
</p>

---

## Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [2.5D Model](#-25d-model)
  - [Architecture](#architecture)
  - [Training](#training-the-25d-model)
  - [Inference](#inference-with-the-25d-model)
- [3D Model](#-3d-model)
  - [Architecture](#architecture-1)
  - [Training](#training-the-3d-model)
  - [Inference](#inference-with-the-3d-model)
- [Ensemble](#-ensemble-model)
- [Results](#results) 

---

## Overview

This model is an ensemble of two complementary segmentation approaches:

| Component | Framework | Architecture | Input |
|-----------|-----------|--------------|-------|
| 2.5D Model | TensorFlow / Keras | Attention Residual U-Net | 2D slices along X, Y, Z planes |
| 3D Model | PyTorch / MONAI | 3D U-Net | 3D patches (64×64×64) |
| Ensemble | — | Averaged probability maps | Full 3D volume |

Both models were trained on first post-contrast DCE-MRI images using structural tumor volume (STV) manual annotations, with 5-fold cross-validation, based on the dataset provided in the paper "Expert tumor annotations and radiomics for locally advanced breast cancer in DCE-MRI for ACRIN 6657/I-SPY1" (https://doi.org/10.1038/s41597-022-01555-4).

---

## Repository Structure

```
.
├── 2d_model/
│   ├── main.py            # Training & inference entry point
│   ├── models.py          # Attention U-Net / Attention ResUNet definitions
│   ├── params.py          # All hyperparameters
│   ├── losses.py          # BCE + Dice loss
│   ├── augmentation.py    # Albumentations pipeline
│   ├── data_gen.py        # Data generator
│   ├── preprocess.py      # Volume slicing & normalization
│   └── postprocess.py     # Volume reconstruction from slices
│
└── 3d_model_monai/
    ├── main_3d.py          # Training & inference entry point
    ├── params_3d.py        # All hyperparameters
    ├── preprocess.py       # Volume loading & normalization
    └── metric.py           # Dice metric utilities
```

---

## Requirements

**2.5D Model**
```bash
pip install tensorflow>=2.x nibabel opencv-python scikit-image albumentations scikit-learn tqdm pandas matplotlib
```

**3D Model**
```bash
pip install torch monai nibabel scikit-learn pandas matplotlib
```

---

## 🔷 2.5D Model

### Architecture

The 2.5D component uses a **Attention Residual U-Net** (`Attention_ResUNet_shallow`) — an encoder–decoder with:

- **5 resolution levels**, base filters = 16 (doubling per level: 16 → 32 → 64 → 128 → 256)
- **Residual convolutional blocks** at each encoder/decoder stage
- **Attention gates** on all skip connections — gating signal from the bottleneck suppresses irrelevant background activations
- **Loss**: combined Binary Cross-Entropy + Dice (`bce_dice_loss`)
- **Output**: sigmoid activation → binary segmentation mask


### Training the 2.5D Model

**Step 1 — Configure paths and hyperparameters**

Edit `2d_model/params.py`:

```python
opts['base_path']   = '/path/to/images/'   # .nii.gz DCE-MRI volumes
opts['base_masks']  = '/path/to/masks/'    # .nii.gz STV annotations
opts['cv_data']     = '/path/to/cv_folds/' # output dir for sliced data
opts['model_save_path']     = '/path/to/save/models/'
opts['results_save_path']   = '/path/to/save/2d_results/'
opts['results_save_path_3d']= '/path/to/save/3d_results/'
```

Key training hyperparameters (already set in `params.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epoch_num` | 60 | Training epochs |
| `batch_size` | 16 | Batch size |
| `crop_size` | 224 | Input slice size (px) |
| `init_LR` | 0.001 | Initial learning rate |
| `LR_decay_factor` | 0.5 | LR halved every `LR_drop_after_nth_epoch` |
| `LR_drop_after_nth_epoch` | 12 | LR schedule step |
| `k_fold` | 5 | Cross-validation folds |
| `treshold` | 0.5 | Binarization threshold |

**Step 2 — Run training**

```bash
cd 2d_model
python main.py
```

The script trains one model that is applied to all three planes (X, Y, and Z planes).


### Inference with the 2.5D Model

Inference is integrated into `main.py` and runs automatically after training each fold. For standalone inference on new volumes:

```python
from preprocess import readImageVolume, normalizeImageIntensityRange
from postprocess import predictVolume
from models import Attention_ResUNet_shallow

# Load model
model = Attention_ResUNet_shallow((224, 224, 3), dropout_rate=0.1, batch_norm=True)
model.load_weights('path/to/Attention_ResUNet_shallow_1.h5')

# Load and normalize volume
img = readImageVolume('path/to/volume.nii.gz', normalize=True)

# Predict — slices along all 3 planes, averages probability maps
prediction = predictVolume(img, model, toBin=True)
# prediction shape: same as input volume, values: 0 or 1
```

**How inference works:**
1. For each plane (X, Y, Z): extract all slices → resize to 224×224 → run through model → resize predictions back
2. Reconstruct three full-volume probability maps
3. Average the three maps → threshold at 0.5 → binary mask

---

## 🔶 3D Model

### Architecture

The 3D component uses **MONAI's standard 3D U-Net** (`monai.networks.nets.UNet`):

- **Spatial dims**: 3
- **Encoder channels**: (16, 32, 64, 128, 256) with stride-2 downsampling
- **Residual units**: 2 per level
- **Normalization**: Batch Normalization
- **Loss**: Dice Loss (with softmax, one-hot labels)
- **Inference**: Sliding window (window = 64×64×64, batch = 4)

### Training the 3D Model

**Step 1 — Configure paths and hyperparameters**

Edit `3d_model_monai/params_3d.py`:

```python
opts['base_path']  = '/path/to/images/'   # .nii.gz DCE-MRI volumes
opts['base_masks'] = '/path/to/masks/'    # .nii.gz STV annotations
opts['model_save_path']      = '/path/to/save/models/'
opts['results_save_path_3d'] = '/path/to/save/predictions/'
```

Key training hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epoch_num` | 100 | Max training epochs |
| `batch_size` | 4 | Batch size (2 volumes × 4 crops) |
| `crop_size` | 64 | 3D patch size (64×64×64) |
| `init_LR` | 0.0001 | Adam learning rate |
| `k_fold` | 5 | Cross-validation folds |

**Step 2 — Run training**

```bash
cd 3d_model_monai
python main_3d.py
```


---

### Inference with the 3D Model

Inference uses **sliding window** to handle arbitrary volume sizes:

```python
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference

device = torch.device("cuda:0")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model.load_state_dict(torch.load('path/to/unet_3d_1.pth'))
model.eval()

with torch.no_grad():
    # input_tensor shape: (1, 1, H, W, D)
    prediction = sliding_window_inference(
        input_tensor, roi_size=(64, 64, 64), sw_batch_size=4, predictor=model
    )
    # prediction shape: (1, 2, H, W, D) — softmax over 2 classes
```
---

## 🔀 Ensemble Model

The final Model prediction combines the outputs of both components.

### How it works

```
3D Volume
    │
    ├──► 2.5D Model (X plane) ──► Prob. Map X ──┐
    ├──► 2.5D Model (Y plane) ──► Prob. Map Y ──┤
    ├──► 2.5D Model (Z plane) ──► Prob. Map Z ──┤──► Average ──► 2.5D Prob. Map ──┐
    │                                                                               │
    └──► 3D Model (sliding window) ──────────────────────────► 3D Prob. Map  ──────┤
                                                                                    │
                                                                              Combine
                                                                                    │
                                                                          Final Segmentation
```
---
### Running the Ensemble

After training both models, combine their predictions:

```python
import numpy as np
import nibabel as nib

# Load 2.5D probability map (output of predictVolume with toBin=False)
prob_2d = np.load('path/to/prob_map_2d.npy')   # shape: (H, W, D), values in [0, 1]

# Load 3D probability map (softmax class-1 channel from sliding_window_inference)
prob_3d = np.load('path/to/prob_map_3d.npy')   # shape: (H, W, D), values in [0, 1]

# Ensemble: average and threshold
ensemble_prob = (prob_2d + prob_3d) / 2.0
ensemble_mask = (ensemble_prob > 0.5).astype(np.uint8)

# Save as NIfTI
ref = nib.load('path/to/original_volume.nii.gz')
nib.save(nib.Nifti1Image(ensemble_mask, ref.affine), 'ensemble_prediction.nii.gz')
```
---
## Results

Predictions and Dice scores of the ensemble model for the derived I-SPY1 dataset (161 samples), as well as the Dice scores of the baseline model presented in [https://doi.org/10.1038/s41597-022-01555-4], are available in the `Results` folder.

---



## 📝 Notes

- All models are evaluated using the **Dice Similarity Coefficient (DSC)**.
- Both pipelines use **5-fold cross-validation** with the same random seed (`19`) for reproducibility.
- Input volumes are **z-score normalized** (bias-corrected, resampled) prior to training.
- The 2.5D model converts single-channel grayscale slices to **3-channel pseudo-RGB** before passing them to the network.
