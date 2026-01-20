import logging
import pathlib
from typing import Iterable, TypedDict

import cv2
import numpy as np
from ultralytics import YOLO  # TODO: Decouple from ultralytics here?

logger = logging.getLogger(__name__)


def load_ultralytics_yolo_model(model_path: pathlib.Path, device: str = "cpu") -> YOLO:
    """
    Load a ultralytics YOLO model at the given path and put it on the specified device.

    Args:
        model_path: The path to the serialized YOLO model.
        device: The device to load the model onto (defaults to CPU).

    Returns:
        The loaded model.
    """
    logger.info("Loading model: %s ...", model_path)
    model = YOLO(model_path)
    logger.info("Model loaded. Sending to device: %s ...", device)
    model.to(device)
    return model


def load_images(paths: Iterable[pathlib.Path]) -> tuple[list[np.ndarray], list[str]]:
    """
    Loads all of the given image paths into numpy arrays.

    Args:
        paths: The path to all of the images to load.

    Returns:
        A tuple of the image arrays and the names of the files that
        they came from for all of the images that were successfully loaded.
    """
    logger.info("Loading %s images ...", len(paths))
    images = []
    names = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning(
                "Unable to load image for %s - no inference will be performed on this image.", p
            )
        images.append(img)
        names.append(p.name)
    logger.info("Successfully loaded %s images.", len(images))
    return images, names


class Detection(TypedDict):
    """A json friendly format for a bounding box prediction."""

    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]


def process_ultralytics_yolo_batched_detections(results) -> list[list[Detection]]:
    """
    Reformats the output of the ultralytics YOLO model into a json friendly format.

    Args:
        results: The output of an ultralytics YOLO model.

    Returns:
        The output processed into our app's API format.
    """

    logger.info("Recieved %s results to process.", len(results))
    batch_detections: list[list[Detection]] = []

    for r in results:
        detections: list[Detection] = []
        for box in r.boxes:
            detections.append(
                {
                    "class_id": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "bbox": tuple(box.xyxy[0].tolist()),
                }
            )
        batch_detections.append(detections)

    return batch_detections
