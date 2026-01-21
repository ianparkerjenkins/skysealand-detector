import io
import logging
import pathlib
import time
from typing import TypedDict

import numpy as np
from fastapi import UploadFile
from PIL import Image

# TODO: Decouple from ultralytics here?
from ultralytics import YOLO  # pyright: ignore [reportPrivateImportUsage]

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


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def _load_and_verify_image_data(data: bytes) -> np.ndarray:
    """Verifies that the image bytes data can be loaded and converts it to a numpy array."""
    image = Image.open(io.BytesIO(data))
    image.verify()
    image = Image.open(io.BytesIO(data))

    if image.format not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported format: {image.format}")

    return np.array(image.convert("RGB"))


def load_images(
    *paths: pathlib.Path | UploadFile, skip_errors: bool = False
) -> tuple[list[np.ndarray], list[str]]:
    """
    Loads all of the given image paths into numpy arrays.

    Args:
        paths: The path to all of the images to load.
        skip_errors: Whether to skip errors with loading images
            and just log a warning instead. Defaults to false.

    Returns:
        A tuple of the image arrays and the names of the files that
        they came from for all of the images that were successfully loaded.
    """
    logger.info("Loading %d images ...", len(paths))

    images: list[np.ndarray] = []
    names: list[str] = []

    for to_load in paths:
        filename = "<unknown-filename>"
        try:
            if isinstance(to_load, pathlib.Path):
                filename = to_load.name
                data = to_load.read_bytes()
            else:  # Should be FastAPI file upload.
                filename = to_load.filename if to_load.filename is not None else filename
                data = to_load.file.read()
            img = _load_and_verify_image_data(data)
        except Exception as e:
            if skip_errors:
                logger.warning("Unable to load image due to error: %s", str(e))
                continue
            raise ValueError(f"Invalid image '{filename}': {e}") from e

        images.append(img)
        names.append(filename)

    logger.info("Successfully loaded %d images.", len(images))
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


class SingleInferenceJsonOutput(TypedDict):
    filename: str
    inference: list[Detection]


class InferenceMetaData(TypedDict):
    num_images: int
    inference_time_sec: float


class InferenceJsonOutput(TypedDict):
    results: list[SingleInferenceJsonOutput]
    metrics: InferenceMetaData


def run_model_with_timing(
    model: YOLO, images: list[np.ndarray], filenames: list[str]
) -> InferenceJsonOutput:
    """
    Runs the given model on the given images and outputs
    the results in a json friendly format.

    The output also contains metadata about this inference,
    e.g. how long it took, how many images were analyzed, etc.

    Args:
        model: The YOLO model to run
        images: The images to input to the model
        filenames: The names of the files that the images
            originated from.

    Returns:
        A json object containing the results of the inference
        and metadata about this inference.
    """
    logger.info("Running inference. This may take a minute ...")
    inference_start = time.perf_counter()
    raw_outputs = model(images)
    outputs = process_ultralytics_yolo_batched_detections(raw_outputs)
    inference_time = time.perf_counter() - inference_start

    response: list[SingleInferenceJsonOutput] = [
        {"filename": name, "inference": output}
        for name, output in zip(filenames, outputs, strict=True)
    ]

    logger.info(
        "Inference complete | images=%d | inference_time=%.3fs",
        len(images),
        inference_time,
    )

    return {
        "results": response,
        "metrics": {
            "num_images": len(images),
            "inference_time_sec": inference_time,
        },
    }
