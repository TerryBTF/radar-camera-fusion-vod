#!/usr/bin/env python
"""Pre-download ResNet18 weights on login node."""

import sys
from torchvision.models import ResNet18_Weights, resnet18

print("Downloading ResNet18 pretrained weights...")
try:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    print("✅ ResNet18 weights downloaded successfully!")
    print("   Cached at: ~/.cache/torch/hub/checkpoints/")
except Exception as e:
    print(f"❌ Failed to download weights: {e}")
    sys.exit(1)
