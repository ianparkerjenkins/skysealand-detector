"""
A utility script for training a simple, baseline YOLO model on the dataset.
"""

# I'm pretty sure this works, not sure why pyright is not happy.
import random

import numpy as np
import torch
from ultralytics import YOLO  # pyright: ignore [reportPrivateImportUsage]


def train():
    # Load pretrained YOLOv8-nano model
    model = YOLO("yolov8n.pt")

    # Set a fixed seed for reproducability.
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Train the model
    model.train(
        data="data/data.yaml",  # adjust to your dataset YAML
        epochs=1,  # ~20–30 epochs
        imgsz=640,  # input size
        batch=16,  # adjust for Colab GPU memory
        workers=2,  # number of data loader workers
        seed=42,  # fixed seed
        pretrained=True,  # use transfer learning
    )

    # Evaluate on validation set
    metrics = model.val()

    print("mAP@0.5:", metrics.box.map50)
    print("Per-class precision:", metrics.box.prec)
    print("Per-class recall:", metrics.box.rec)


if __name__ == "__main__":
    train()
