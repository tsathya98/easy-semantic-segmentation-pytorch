# Semantic Segmentation PyTorch

This repository provides an implementation of semantic segmentation models using PyTorch.

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Models](#models)
- [Training](#training)
  - [Optimization](#optimization) 
  - [Loss Functions](#loss-functions)
  - [Augmentation](#augmentation)
  - [Monitoring](#monitoring) 
  - [Checkpointing](#checkpointing)
- [Inference](#inference)
- [Evaluation](#evaluation)  
- [Visualization](#visualization)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [To Do](#to-do)
- [References](#references)

## Installation

The code has been tested with the following versions:

- Python 3.9
- PyTorch 1.10
- CUDA 11.3

To install dependencies:

```bash
# Create conda environment 
conda create -n semantic-segmentation python=3.9
conda activate semantic-segmentation

# Install PyTorch and OpenCV
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Datasets

The implementation supports common semantic segmentation datasets like:

- [Cityscapes](https://www.cityscapes-dataset.com/)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [COCO](https://cocodataset.org/#home)
- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

The dataset needs to be divided into train, validation and test sets for training.

## Models

The repository provides several state-of-the-art semantic segmentation models:

- UNet
- DeepLabV3
- DeepLabV3+
- PSPNet
- PAN
- FPN
- Unet++
- MANet

These models use encoders like MobileNet, ResNet, ResNeXt, EfficientNet, DarkNet etc. Pre-trained weights from ImageNet, Noisy-student, Advprop or other self-supervised methods can be used for initialization.

The model architecture and encoder can be specified in `config.yml`.

## Training

To start training, update the dataset paths and other parameters in `config.yml`:

```yaml
# config.yml

dataset:
  train_images_dir: "/path/to/train/images/"
  train_masks_dir: "/path/to/train/masks/"
  
  val_images_dir: "/path/to/val/images/"
  val_masks_dir: "/path/to/val/masks/"

model_name: "DeepLabV3Plus"
encoder: "resnet34" 
encoder_weights: "imagenet" 

# Other parameters  
```

Then run:

```bash
python train.py
``` 

Training progress can be monitored using Weights & Biases.

### Optimization

The model is trained using AdamW optimizer with OneCycleLR scheduling by default. These can be configured in `config.yml`.

### Loss Functions

The implementation provides several loss functions like:

- CrossEntropyLoss
- DiceLoss 
- BinaryCrossEntropy + DiceLoss
- LovaszSoftmaxLoss
- TverskyLoss
- RMILoss

The loss function can be specified in `config.yml`.


### Augmentation

Data augmentation techniques like random flip, rotate, crop, color jitter etc. can be added to prevent overfitting. This is handled by Albumentations.

### Monitoring 

Training progress can be monitored using TensorBoard or Weights & Biases. These can be configured in `config.yaml`.

### Checkpointing

Model checkpoints are saved during training for resuming. The frequency can be configured in `config.yaml`.


## Inference 

The trained model can be used to get predictions on new images:

```python
import torch

model = torch.load("model.pth")

# Load image
img = ... 

# Get prediction
output = model(img)
```

Update the model path and load images in `predict.py`.


## Evaluation

Model performance is evaluated using IoU (Intersection over Union) metric. IoU scores on train and validation sets are logged during training.

Other metrics like pixel-wise accuracy, Dice coefficient can also be used.


## Visualization

The predictions can be visualized as masks overlaid on the input images using OpenCV or Matplotlib.

## Deployment

The trained PyTorch model can be optimized and deployed using ONNX Runtime, TensorFlow Lite or TensorFlow Serving for inference in production.

## Contributing

Contributions to add new models, datasets, augmentation techniques etc. are welcome!

## To Do

| Task | Done | 
|-|-|  
| Support additional optimizers via config | ⬜ |
| Support more LR schedulers via config | ⬜ |
| Support ADE20K dataset | ⬜ |
| Support Cityscapes dataset | ⬜ | 
| Support COCO Stuff dataset | ⬜ |
| Support PASCAL VOC 2012 dataset | ⬜ |
| Add test time augmentation | ⬜ |
| Add option for diff training & val image sizes | ⬜ |
| Implement early stopping | ⬜ |
| Add mixed precision training | ⬜ |
| Containerize with Docker | ⬜ |

## References

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) 
- [Albumentations](https://albumentations.ai/)
- [Weights & Biases](https://wandb.ai/site)
- [timm](https://github.com/rwightman/pytorch-image-models)

Let me know if any sections need more explanation or details!
