# Geometric Unsupervised Domain Adaptation (GUDA)

This repository contains a PyTorch implementation for **Geometric Unsupervised Domain Adaptation (GUDA)**, a method designed to bridge the sim-to-real domain gap in semantic segmentation.

GUDA leverages self-supervised monocular depth estimation as a proxy task. By combining synthetic semantic supervision with real-world geometric constraints (learned from unlabeled videos), it learns a domain-invariant representation that improves semantic segmentation performance on real-world data without requiring real-world semantic labels.

## 📄 Reference Paper

**Geometric Unsupervised Domain Adaptation for Semantic Segmentation**
*Authors: Vitor Guizilini, Adrien Gaidon, Jie Li, Rareş Ambrus*
*Toyota Research Institute (TRI)*
[arXiv:2103.16694](https://arxiv.org/abs/2103.16694)

## 📂 Project Structure

```bash
geometric_unsupervised_domain_adaptation/
├── cfg/
│   └── monodepth.yaml      # Configuration file for training hyperparameters
├── dataset/
│   ├── carla_dataset.py    # Dataloader for synthetic CARLA data (Source Domain)
│   ├── kitti.py            # Dataloader for real-world KITTI data (Target Domain)
│   └── data_utils.py       # Data augmentation and preprocessing utilities
├── layers/                 # Neural Network Modules
│   ├── encoder.py          # ResNet-based Encoder
│   ├── depth_decoder.py    # Depth estimation head
│   ├── semantic_decoder.py # Semantic segmentation head
│   ├── pose_decoder.py     # Pose estimation for SfM (Structure from Motion)
│   ├── posenet.py          # Pose network wrapper
│   └── layers.py           # Custom layers (SSIM, 3D projection, etc.)
├── train.py                # Main training script
├── utils.py                # Helper functions (logging, visualization)
├── test.ipynb              # Jupyter notebook for testing and visualization
└── README.md

```

## 🛠️ Installation

### Prerequisites

* Python 3.7+
* PyTorch (tested on 1.7+)
* CUDA (for GPU acceleration)

### Dependencies

Install the required Python packages:

```bash
pip install torch torchvision numpy opencv-python pyyaml matplotlib pillow

```

## 💾 Dataset Preparation

This project requires two datasets: a **Source (Synthetic)** domain and a **Target (Real)** domain.

### 1. Source Domain: CARLA (Synthetic)

The code expects synthetic data generated from the CARLA simulator, including RGB images, depth maps, and semantic segmentation masks.

* **Structure:**
```
/path/to/carla_data/
├── rgb_images/       # RGB frames
├── depth/            # Depth maps
└── semantic_mask/    # Semantic segmentation masks

```


* **Configuration:** Update `carla_path` in `cfg/monodepth.yaml`.

### 2. Target Domain: KITTI (Real)

The code uses the KITTI dataset for the target domain, utilizing the raw data format for self-supervised depth learning.

* **Download:** [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php)
* **Structure:**
```
/path/to/kitti_data/
├── 2011_09_26/
├── 2011_09_28/
...

```


* **Configuration:** Update `data_path` in `cfg/monodepth.yaml`.

## ⚙️ Configuration

Training parameters are controlled via `cfg/monodepth.yaml`. Key parameters include:

```yaml
data:
  path: "path/to/kitti"
  carla_path: "path/to/carla"
  image_size: [192, 640]  # Input resolution (H, W)
  batch_size: 12

model:
  encoder_name: "resnet18"
  pretrained: true        # Use ImageNet pretrained weights

training:
  epochs: 20
  learning_rate: 1e-4

```

## 🚀 Usage

### Training

To train the model using both synthetic supervision and real-world geometric self-supervision:

```bash
python train.py --config cfg/monodepth.yaml --device cuda

```

**Arguments:**

* `--config`: Path to the YAML config file (default: `cfg/monodepth.yaml`).
* `--resume`: Path to a checkpoint to resume training from.
* `--device`: Device to run on (`cuda` or `cpu`).

### Testing & Visualization

A Jupyter Notebook is provided for testing the model and visualizing results (Depth maps, Segmentation masks, and Reconstructions).

```bash
jupyter notebook test.ipynb

```

## 🧠 Model Architecture

The architecture consists of a shared **Encoder** (ResNet-18) that feeds into task-specific **Decoders**:

1. **Depth Decoder**: Predicts dense depth maps (trained via photometric consistency on KITTI and supervised depth on CARLA).
2. **Semantic Decoder**: Predicts semantic segmentation masks (trained via supervised labels on CARLA).
3. **Pose Network**: Predicts ego-motion between frames (used for the self-supervised geometric loss on KITTI).

This multi-task setup allows the model to learn geometric features from unlabeled real data that are robust enough to transfer semantic knowledge from the synthetic domain.

## 🤝 Citation

If you use this code or method in your research, please cite the original paper:

```bibtex
@article{guizilini2021geometric,
  title={Geometric Unsupervised Domain Adaptation for Semantic Segmentation},
  author={Guizilini, Vitor and Gaidon, Adrien and Li, Jie and Ambrus, Rare{\c{s}}},
  journal={arXiv preprint arXiv:2103.16694},
  year={2021}
}

```