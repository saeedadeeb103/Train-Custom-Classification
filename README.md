# Train-Custom-Classification

# Your Project Name

A brief and catchy description of your project.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This project is an implementation of a PyTorch-based image classification model using various encoder architectures from the Timm library. The model is designed for flexibility, allowing users to easily fine-tune a pre-trained model for a specific number of classes in their dataset. 

## Features

- Utilizes popular encoder architectures like ResNet, MobileNetV2, and EfficientNet from the Timm library.
- Supports customization of the final classification layer to match the number of classes in your dataset.
- Implements training, validation, and testing steps with PyTorch Lightning, making it easy to train and evaluate the model.

## Getting Started

Instructions on how to get the project up and running on a local machine.

### Prerequisites

- Python 3.8.16
- Pytorch-lightning
- PyTorch
- timm
- hydra-core
- Other dependencies specified in `requirements.txt`

### Installation

```bash
conda create -n ml python==3.8.16
conda activate ml
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
