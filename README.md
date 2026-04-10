# A007 Framework: Fundus Disease Segmentation with Dual-Channel Architecture

A comprehensive deep learning framework for automated fundus disease segmentation, optimized for multi-disease detection in retinal imaging. This project explores advanced neural network architectures and multi-modal input strategies for improved diagnostic accuracy.

## Key Contributions

### 1. **Dual-Channel Input Architecture**
We systematically investigate the impact of different input channel combinations on segmentation accuracy:
- **Monocular Processing**: Single-channel input from left or right eye
- **Color Merge**: Concatenated RGB channels from both eyes
- **Output Merge**: Separate processing with output-level fusion
- **Network-Level Merge**: Dual-stream networks with integrated fusion
- **Double Merge**: Dual independent models with learned fusion head

Our experiments demonstrate that strategic multi-channel fusion significantly improves detection of subtle pathological features.

### 2. **DeepLabV3+ Optimization**
Enhanced DeepLabV3+ architecture specifically tailored for fundus imaging:
- Improved ASPP (Atrous Spatial Pyramid Pooling) modules
- Spatial attention mechanisms for disease localization
- Attention-based classification heads for multi-disease prediction
- Cross-field transformer blocks for capturing long-range dependencies

### 3. **Data Augmentation for Medical Imaging**
Specialized augmentation strategies for fundus disease datasets:
- Rotation, scaling, and elastic deformations
- Color jittering and contrast adjustment
- Synthetic data generation for rare disease classes
- Balanced sampling across disease categories

### 4. **Multi-Disease Classification**
Support for 8 fundus disease categories:
- **N**: Normal
- **A**: Age-related Macular Degeneration (AMD)
- **C**: Cataract
- **D**: Diabetic Retinopathy (DR)
- **G**: Glaucoma
- **H**: Hypertensive Retinopathy
- **M**: Myopia
- **O**: Other diseases

## Project Structure

```
A007_framework/
├── models/                          # Model implementations
│   ├── deeplabv3+/                 # DeepLabV3+ variants
│   ├── resnet_img_merge/           # ResNet with image-level fusion
│   ├── resnet_double/              # Dual ResNet architecture
│   ├── visiontransformer/          # Vision Transformer models
│   ├── cross_field_transformer/    # Cross-field attention models
│   └── load.py                     # Model loading utilities
├── networks/                        # Network building blocks
│   ├── deeplabv3plus.py            # DeepLabV3+ implementation
│   ├── cross_field_transformer.py  # Transformer modules
│   ├── resnet_color_merge.py       # Color merge ResNet
│   ├── swin_transformer.py         # Swin Transformer
│   └── visiontransformer.py        # Vision Transformer
├── blocks/                          # Reusable components
│   ├── aspp/                       # ASPP modules
│   ├── activation.py               # Custom activations
│   ├── conv.py                     # Convolution layers
│   └── head.py                     # Classification heads
├── dataset/                         # Data loading and preprocessing
│   └── transform/                  # Data augmentation
├── loss/                            # Loss functions
├── metrics/                         # Evaluation metrics
├── optims/                          # Optimizers
├── tools/                           # Training and validation
│   ├── train.py                    # Training loops
│   └── val.py                      # Validation functions
├── lhy_tools/                       # Utility tools
│   ├── data_classify.py            # Data classification
│   ├── data_splite.py              # Train/val/test split
│   ├── delete_black.py             # Remove invalid images
│   ├── new_image_trans.py          # Image transformation
│   ├── show_ckp.py                 # Model checkpoint visualizer
│   ├── preprocess_analysis.py      # Data analysis
│   └── tensor_view/                # Visualization utilities
├── visualization/                   # Real-time training monitoring
│   └── app.py                      # Web-based dashboard
└── roi/                             # Region of interest processing
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy, scipy
- tqdm
- einops

### Setup
```bash
git clone <repository-url>
cd A007_framework
pip install -r requirements.txt
```

## Usage

### Training

#### Basic Training
```python
from tools.train import train_model
from models.load import load_model

model = load_model('deeplabv3+')
train_model(
    model=model,
    train_loader=train_loader,
    loss_fn=loss_function,
    optimizer=optimizer,
    visualizer=visualizer,
    num_epochs=100,
    val=True,
    val_loader=val_loader,
    metric=metric
)
```

#### Dual-Channel Training
```python
from tools.train import train_color_merge_model

train_color_merge_model(
    model=model,
    train_loader=train_loader,  # Provides (left_img, right_img, label)
    loss_fn=loss_function,
    optimizer=optimizer,
    visualizer=visualizer,
    num_epochs=100,
    val=True,
    val_loader=val_loader,
    metric=metric
)
```

#### Double Merge (Dual Models)
```python
from tools.train import train_double_merge_model

train_double_merge_model(
    model_1=model_left,
    model_2=model_right,
    head=fusion_head,
    train_loader=train_loader,
    loss_fn=loss_function,
    optimizer_1=optimizer1,
    optimizer_2=optimizer2,
    visualizer=visualizer,
    num_epochs=100,
    val=True,
    val_loader=val_loader,
    metric=metric
)
```

### Validation

```python
from tools.val import val_model

metrics = val_model(
    model=model,
    val_loader=val_loader,
    metric=metric,
    device='cuda'
)
```

### Real-Time Monitoring

Launch the web-based visualization dashboard:
```bash
cd visualization
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000` to monitor training progress in real-time.

## Utility Tools

### Data Processing

**Data Classification** (`lhy_tools/data_classify.py`)
- Organize images by disease category
- Create structured dataset folders

**Data Split** (`lhy_tools/data_splite.py`)
- Split dataset into train/validation/test sets
- Maintain class balance across splits

**Image Cleaning** (`lhy_tools/delete_black.py`)
- Remove corrupted or invalid images
- Filter out low-quality fundus images

**Image Transformation** (`lhy_tools/new_image_trans.py`)
- Apply preprocessing to raw fundus images
- Normalize and standardize image formats

### Model Analysis

**Checkpoint Visualizer** (`lhy_tools/show_ckp.py`)
- GUI tool to inspect model weights
- Compare model architecture with saved checkpoints
- Verify weight compatibility before loading

**Log Reader** (`lhy_tools/read_log.py`)
- Parse training logs
- Extract metrics and statistics

**Checkpoint Renaming** (`lhy_tools/rename_checkpoint.py`)
- Batch rename model checkpoints
- Organize saved models

### Visualization

**Tensor Visualization** (`lhy_tools/tensor_view/`)
- `heat_view.py`: Heatmap visualization of attention maps
- `gray_view.py`: Grayscale visualization of feature maps

**Preprocessing Analysis** (`lhy_tools/preprocess_analysis.py`)
- Analyze dataset statistics
- Visualize data distribution

## Model Architectures

### DeepLabV3+ Variants
- Standard DeepLabV3+ with attention heads
- Spatial attention-enhanced version
- Output merge variant for dual-channel input

### ResNet-Based Models
- ResNet50 with color merge
- ResNet50 with output merge
- ResNet50 with FPN (Feature Pyramid Network)
- ResNet50 with attention heads

### Transformer-Based Models
- Vision Transformer (ViT)
- Swin Transformer
- Cross-Field Transformer (custom architecture)

### Hybrid Architectures
- ResNet + Transformer fusion
- Multi-head disease classifiers
- Dual-stream networks with learned fusion

## Training Configuration

The framework supports flexible training configurations through model filenames encoding hyperparameters:

```
model_name_[input_strategy]_[loss]_[size]_[loss_fn]_[optimizer]_[lr]_[bs].py
```

Example: `deeplabv3plus_color_merge_ce_attention_head_512_bce_adam_lr1e-3_bs32.py`
- Input: Color merge (dual-channel concatenation)
- Loss: Cross-entropy + BCE
- Size: 512×512 images
- Optimizer: Adam with lr=1e-3
- Batch size: 32

## Evaluation Metrics

The framework computes comprehensive metrics across multiple thresholds:
- **Accuracy**: Pixel-level classification accuracy
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall

Metrics are computed for each disease class and aggregated across thresholds for robust evaluation.

## Key Features

✅ **Multi-Architecture Support**: Seamlessly switch between different model architectures
✅ **Flexible Input Strategies**: Experiment with various dual-channel fusion approaches
✅ **Real-Time Monitoring**: Web-based dashboard for training visualization
✅ **Comprehensive Tooling**: Data processing, model analysis, and visualization utilities
✅ **Medical Imaging Optimized**: Specialized augmentation and preprocessing for fundus images
✅ **Multi-Disease Support**: Simultaneous detection of 8 disease categories
✅ **Modular Design**: Easy to extend with new architectures and components

## References & Acknowledgments

This project builds upon and references the following works:

- **DeepLabV3+**: Chen, L. C., et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." ECCV 2018.
- **Vision Transformer**: Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- **Swin Transformer**: Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
- **Attention Mechanisms**: Woo, S., et al. "CBAM: Convolutional Block Attention Module." ECCV 2018.

We acknowledge the contributions of the medical imaging and computer vision communities in advancing fundus disease detection methodologies.

## Contact

For questions, suggestions, or collaboration inquiries, please reach out:

**Email**: [your-email@example.com]

---

**Last Updated**: April 2026
**License**: [Specify your license here]
