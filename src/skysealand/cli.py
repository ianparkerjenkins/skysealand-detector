"""
The command line interface for the project.
"""

import typer

from skysealand import logging_setup
from skysealand.dataset import download as data_download
from skysealand.dataset import validation
from skysealand.train import yolo_baseline

app = typer.Typer()


@app.command()
def download():
    """Download the dataset"""
    logging_setup.setup_logging()

    data_download.download()
    data_download.extract()
    validation.validate_all_data()


@app.command()
def train():
    """Train the model"""
    logging_setup.setup_logging()

    yolo_baseline.train()
