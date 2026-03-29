# Event-based Face Liveness Detection

This repository provides a minimal and practical pipeline to train a
**Spiking Convolutional Neural Network (SCNN)** for **event-based face
liveness detection** using **SpikingJelly**.

------------------------------------------------------------------------

## Overview

This project focuses on **binary classification** for face
anti-spoofing:

-   `0` → Fake / Spoof\
-   `1` → Real / Live

The model operates on raw event streams stored as `.npy` files and
converts them into dense spatio-temporal tensors.

------------------------------------------------------------------------

## Input Format

Each sample is stored as:

    [N, 4] = (x, y, t, p)

Where:

-   `x, y`: pixel coordinates\
-   `t`: timestamp\
-   `p`: polarity (-1 or +1)

------------------------------------------------------------------------

## Project Structure

    scnn/
    ├── train.py
    ├── dataset.py
    ├── model.py
    └── utils.py

------------------------------------------------------------------------

## Installation

Install dependencies:

    pip install torch torchvision spikingjelly numpy tqdm scikit-learn

------------------------------------------------------------------------

## Dataset Structure

    data/
    ├── train/
    │   ├── fake/
    │   └── real/
    ├── val/
    │   ├── fake/
    │   └── real/

Folder names are used as labels:

-   `fake` → 0\
-   `real` → 1

------------------------------------------------------------------------

## Event Representation

Raw events are converted into a dense tensor:

    [T, 2, H, W]

Where:

-   `T`: number of temporal bins (recommended: 8)
-   `2`: polarity channels
-   `H, W`: spatial resolution

------------------------------------------------------------------------

## Model

The architecture is a **Spiking CNN (SCNN)** composed of:

-   Convolutional layers
-   Batch Normalization
-   LIF spiking neurons
-   Temporal aggregation (mean over time)

------------------------------------------------------------------------

## Training

Run training with:

    python train.py

The training pipeline:

-   Loads `.npy` event files
-   Converts events → tensor
-   Trains SCNN
-   Evaluates on validation set
-   Saves best model based on **ACER**

------------------------------------------------------------------------

## Metrics

This project uses **biometric evaluation metrics**:

-   Accuracy
-   APCER (Attack error)
-   BPCER (Bona fide error)
-   ACER (primary metric)
-   AUC-ROC

ACER is used as the main metric for model selection.

------------------------------------------------------------------------

## Notes

-   Model input: `[T, B, C, H, W]`

-   Always reset spiking states:

        functional.reset_net(model)

-   Use GPU for better performance

------------------------------------------------------------------------

## Author

- Nicolas Mastropasqua (Universidad de Buenos Aires, Argentina)
- Ignacio Bugueno-Cordova (Chile)
- Rodrigo Verschae (Universidad Técnica Federico Santa María, Chile)

