"""
A utility script for training a simple, baseline YOLO model on the dataset.
"""

import logging
import random

import numpy as np
import torch
from ultralytics import YOLO  # pyright: ignore [reportPrivateImportUsage]

logger = logging.getLogger(__name__)


def train():
    # Load pretrained YOLOv8-nano model
    model = YOLO("yolov8n.pt")

    # Set a fixed seed for reproducability.
    seed = 42
    torch.manual_seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Train the model
    model.train(
        data="data/data.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        workers=2,
        seed=seed,
        pretrained=True,
    )

    # Evaluate on validation set
    metrics = model.val()

    logger.info("mAP@0.5:, %s", metrics.box.map50)
    logger.info("Per-class precision: %s", metrics.box.prec)
    logger.info("Per-class recall: %s", metrics.box.rec)
