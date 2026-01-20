"""
The command line interface for the project.
"""

import json
import logging
import pathlib

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


@app.command()
def infer(images: list[str], model_path: str = "yolov8n.pt", output_path: str = "inference.json"):
    """
    Run inference on a batch of image paths.

    Writes the result to the specified ``output_path`` location as a json file.
    This file has the structure of ``{'filename': ..., 'inference': ...}``.

    Args:
        images: A list of the image paths to perform inference on.
        model_path: The path to the model to use for inference. Defaults to "yolov8n.pt"
            (The result of performing the default training).
        output_path: The path to the output file to write. Defaults to "inference.json"
    """
    logging_setup.setup_logging()

    image_arrays, names = inference.load_images([pathlib.Path(img) for img in images])

    model = inference.load_ultralytics_yolo_model(model_path)
    outputs = inference.process_ultralytics_yolo_batched_detections(model(image_arrays))

    to_write: dict[str, list[list[inference.Detection]]] = {}
    for name, output in zip(names, outputs, strict=True):
        to_write["filename"] = name
        to_write["inference"] = output

    logger.info("Writing output file @ %s ...", output_path)
    with pathlib.Path.open(output_path, "w") as write_file:
        json.dump(to_write, write_file)
    logger.info("Done with inference!")
