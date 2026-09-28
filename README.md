# Sugarcane

# Sugarcane Weed Segmentation Pipeline

Official repository for the paper: **"A Methodological Framework for Weed Segmentation in Sugarcane Fields Using YOLOv8, SAHI, and Spatial Morphological Filtering"**.

## Overview

This repository contains the inference pipeline designed for high-resolution orthomosaic weed detection and segmentation in sugarcane crops. The method integrates:
1. **YOLOv8-seg** for instance segmentation.
2. **Slicing Aided Hyper Inference (SAHI)** to handle high-resolution imagery without spatial loss (slice size: 1024 x 1024 px, 20% overlap).
3. **Spatial Morphological Filter** ($\lambda_{max} = 0.10$) to reduce false-positive detections caused by canopy closure.

## Repository Structure

```text
├── src/
│   ├── inference.py          # Main execution script
│   └── spatial_filter.py     # Spatial Morphological Filter module
├── requirements.txt          # Environment dependencies
└── README.md                 # Project documentation
