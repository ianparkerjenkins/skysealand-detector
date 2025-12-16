"""
Downloads the relevant SkySeaLand dataset if it's not already present.

If this is run and the dataset already exists, then the data will not be redownloaded.
"""

import logging
import pathlib
import urllib.request
import zipfile

logger = logging.getLogger(__name__)

DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/mdzahidhasanriad/skysealand"
DATA_DIR = pathlib.Path("data")
ZIP_PATH = DATA_DIR / "skysealand.zip"


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        logger.info("Dataset already downloaded.")
        return

    logger.info("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    logger.info("Download complete.")


def extract():
    extract_path = DATA_DIR / "SkySeaLand"

    if extract_path.exists():
        logger.info("Dataset already extracted.")
        return

    logger.info("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    logger.info("Extraction complete.")
