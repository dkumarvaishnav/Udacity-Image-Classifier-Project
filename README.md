# Image Classifier Project

A deep learning image classifier built with PyTorch for recognizing 102 different flower species. This project demonstrates transfer learning, neural network architecture design, and end-to-end machine learning pipeline development.

## Table of Contents
- [Project Overview](#project-overview)
- [Performance Metrics](#performance-metrics)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Author](#author)

## Project Overview

This project implements a convolutional neural network (CNN) for flower species classification using transfer learning with VGG16. The model is trained on the Oxford 102 Category Flower Dataset and achieves 81% accuracy on test data.

**Key Features:**
- Transfer learning with pre-trained VGG16 backbone
- Custom classifier head for 102-class classification
- Data augmentation for improved generalization
- GPU/CPU compatible training and inference
- Command-line interface for training and prediction
- Model checkpoint saving and loading

## Performance Metrics

**Test Accuracy:** 81%
**Validation Accuracy:** 70-85% (typical range)
**Training Time:** ~5 minutes on GPU, ~30 minutes on CPU
**Model Size:** ~500MB checkpoint file
**Inference Speed:** ~100ms per image on CPU

**Optimizations Implemented:**
- Transfer learning: Only 2M parameters trained vs. full 138M VGG16 parameters
- Frozen feature extractor for 10x faster training
- Memory-efficient gradient computation
- Data augmentation: RandomRotation, RandomResizedCrop, RandomHorizontalFlip
- Dropout regularization (50%) to prevent overfitting
- Batch processing (batch size: 64) for efficient GPU utilization

## Dependencies

The project requires Python 3 and the following libraries:

- PyTorch
- torchvision
- NumPy
- Matplotlib
- Pillow (PIL)
- Seaborn
- argparse
- json

**Installation:**

```bash
pip install torch torchvision numpy matplotlib pillow seaborn
```

## Dataset

The project uses the [Oxford 102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

**Dataset Statistics:**
- 102 flower species
- 40-258 images per species
- ~6,500 training images
- ~800 validation images
- ~800 test images
- Total size: 328MB

**Download Dataset:**

```bash
python download_dataset.py
```

The `cat_to_name.json` file contains mappings from category labels to flower names.

## Model Architecture

**Base Model:** VGG16 (pre-trained on ImageNet)

**Custom Classifier:**
```
Input: 25,088 features (VGG16 output)
  ↓
Linear(25088 → 120) + ReLU + Dropout(0.5)
  ↓
Linear(120 → 90) + ReLU
  ↓
Linear(90 → 70) + ReLU
  ↓
Linear(70 → 102) + LogSoftmax
```

**Loss Function:** Negative Log Likelihood Loss (NLLLoss)
**Optimizer:** Adam with configurable learning rate

**Data Processing:**
- Training: Random augmentation + normalization (ImageNet statistics)
- Inference: Resize(256) → CenterCrop(224) → Normalize
- Image size: 224×224 pixels

## Training the Model

The `train.py` script trains the image classifier with configurable hyperparameters.

**Command-Line Arguments:**

- `--arch`: Model architecture (default: "vgg16")
- `--save_dir`: Checkpoint save path (default: "./checkpoint.pth")
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--hidden_units`: Hidden layer size (default: 120)
- `--epochs`: Number of training epochs (default: 1)
- `--gpu`: Device to use - "gpu" or "cpu" (default: "gpu")

**Basic Training:**

```bash
python train.py
```

**Advanced Training:**

```bash
python train.py --arch vgg16 --epochs 20 --learning_rate 0.001 --hidden_units 512 --save_dir ./my_model.pth --gpu gpu
```

**Training Process:**
1. Loads and preprocesses the flower dataset
2. Initializes VGG16 with frozen feature layers
3. Trains custom classifier on flower categories
4. Validates on validation set after each epoch
5. Saves trained model checkpoint
6. Tests on test set and reports final accuracy

## Making Predictions

The `predict.py` script performs inference on flower images using a trained model.

**Command-Line Arguments:**

- `--image`: Path to input image (required)
- `--checkpoint`: Path to model checkpoint (required)
- `--top_k`: Number of top predictions to display (default: 5)
- `--category_names`: Path to category-to-name mapping JSON (default: "cat_to_name.json")
- `--gpu`: Device to use - "gpu" or "cpu" (default: "gpu")

**Basic Prediction:**

```bash
python predict.py --image flowers/test/1/image_06743.jpg --checkpoint checkpoint.pth
```

**Top-K Predictions:**

```bash
python predict.py --image flower.jpg --checkpoint checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu cpu
```

**Output Example:**
```
Top 5 Predictions:
1. pink primrose (92.3%)
2. hard-leaved pocket orchid (4.1%)
3. alpine sea holly (1.8%)
4. orange dahlia (0.9%)
5. ruby-lipped cattleya (0.5%)
```

## Testing

Verify all functionality with the test suite:

```bash
python test_functionality.py
```

**Tests Include:**
- Model creation and architecture validation
- Data transform pipeline verification
- Device detection (CPU/GPU)
- Category mapping loading
- Checkpoint structure validation
- Forward pass execution

## Project Structure

```
Udacity-Image-Classifier-Project/
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── test_functionality.py         # Comprehensive test suite
├── download_dataset.py           # Dataset download utility
├── workspace-utils.py            # Workspace utility functions
├── cat_to_name.json             # Category to flower name mapping
├── Image Classifier.ipynb       # Jupyter notebook (development)
├── checkpoint.pth               # Trained model checkpoint
├── PROJECT_STATUS.md            # Project status and fixes
├── DATASET_INFO.md              # Dataset information
└── README.md                    # This file
```

## Author

Dkuma
