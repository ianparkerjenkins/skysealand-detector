import os
import zipfile
import urllib.request
from pathlib import Path

DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/mdzahidhasanriad/skysealand"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "skysealand.zip"

def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print("Dataset already downloaded.")
        return

    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print("Download complete.")

def extract():
    extract_path = DATA_DIR / "SkySeaLand"

    if extract_path.exists():
        print("Dataset already extracted.")
        return

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    print("Extraction complete.")

if __name__ == "__main__":
    download()
    extract()
