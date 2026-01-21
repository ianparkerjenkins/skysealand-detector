"""
The command line interface for the project.
"""

import json
import logging
import pathlib
from typing import TypedDict

import typer

from skysealand import inference, logging_setup
from skysealand.dataset import download as data_download
from skysealand.dataset import validation
from skysealand.train import yolo_baseline

logger = logging.getLogger(__name__)


app = typer.Typer()


@app.command()
def download():
    """Download the dataset"""
    logging_setup.setup_logging()

    data_download.download()
    data_download.extract()
    validation.validate_all_data()
    logger.info("Done with download!")


@app.command()
def train():
    """Train the model"""
    logging_setup.setup_logging()

    yolo_baseline.train()
    logger.info("Done with training!")


class InferenceJsonOutput(TypedDict):
    filename: str
    inference: list[inference.Detection]


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _get_image_paths_from_directory(images_dir: str) -> list[pathlib.Path]:
    dir_path = pathlib.Path(images_dir)
    if not dir_path.is_dir():
        raise ValueError(f"{images_dir} is not a directory")

    image_paths = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    return image_paths


@app.command()
def infer(
    images: list[str] = typer.Argument(
        None,
        help="Image paths to perform inference on",
    ),
    images_dir: str | None = typer.Option(
        None,
        "--images-dir",
        help="Directory containing images to perform inference on",
    ),
    model_path: str = "yolov8n.pt",
    output_path: str = "inference.json",
):
    """
    Run inference on a batch of image paths.

    Writes the result to the specified ``output_path`` location as a json file.
    This file has the structure of ``{'filename': ..., 'inference': ...}``.
    See ``inference.Detection`` for more details.

    Args:
        images: A list of image paths to perform inference on.
        images_dir: A directory containing images to perform inference on.
        model_path: The path to the model to use for inference. Defaults to "yolov8n.pt"
            (The result of performing the default training).
        output_path: The path to the output file to write. Defaults to "inference.json"
    """
    logging_setup.setup_logging()

    if images is None and images_dir is None:
        raise ValueError("You must provide either `images` or `images_dir`.")

    if images is not None and images_dir is not None:
        raise ValueError("Provide only one of `images` or `images_dir`, not both.")

    image_paths = (
        [pathlib.Path(img) for img in images]
        if images_dir is None
        else _get_image_paths_from_directory(images_dir)
    )
    if not image_paths:
        raise ValueError("No images found to process.")

    image_arrays, names = inference.load_images(*image_paths)

    model = inference.load_ultralytics_yolo_model(pathlib.Path(model_path))
    logger.info("Running inference. This may take a minute ...")
    outputs = inference.process_ultralytics_yolo_batched_detections(model(image_arrays))

    to_write: list[InferenceJsonOutput] = []
    for name, output in zip(names, outputs, strict=True):
        to_write.append({"filename": name, "inference": output})

    logger.info("Writing output file @ %s ...", output_path)
    with pathlib.Path(output_path).open("w") as write_file:
        json.dump(to_write, write_file, indent=4)
    logger.info("Done with inference!")
