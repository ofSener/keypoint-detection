# 🔑 Keypoint Detection with Deep Learning

Train a deep learning model to detect and describe keypoints in images, using classical OpenCV methods (SIFT) for pseudo-labels.

## 🚀 Quick Start

### Local Training
```bash
# Clone repository
git clone https://github.com/ofSener/keypoint-detection.git
cd keypoint-detection

# Create conda environment
conda create -n keypoint python=3.10
conda activate keypoint

# Install dependencies
pip install torch torchvision opencv-python tqdm

# Prepare data directories
mkdir -p data/{raw,processed,kp_desc}

# Download and extract dataset
# ... dataset instructions ...

# Preprocess SIFT features
python scripts/preprocess_sift.py --workers 8 --chunk_size 64 --max_kp 300

# Train model
python scripts/train.py train --data data/processed --bs 32 --ep 12 --lr 2e-4
```

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ofSener/keypoint-detection/blob/main/keypoint_training.ipynb)

1. Open the notebook in Colab
2. Enable GPU runtime
3. Run all cells

## 📁 Project Structure
```
keypoint-detection/
├── data/
│   ├── raw/          # Raw dataset files
│   ├── processed/    # Processed images
│   └── kp_desc/      # SIFT keypoint descriptors
├── scripts/
│   ├── preprocess_sift.py  # SIFT feature extraction
│   └── train.py      # Model training
└── keypoint_training.ipynb  # Colab notebook
```

## 🔧 Model Architecture
- Backbone: ResNet18
- Heads:
  - Keypoint heatmap (1 channel)
  - Descriptor (128-dim)

## 📊 Performance
- Training time:
  - CPU: ~3 days
  - Apple Silicon: ~1 day
  - GPU (Colab): 2-3 hours

## 📝 License
MIT 