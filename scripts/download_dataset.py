# scripts/download_dataset.py

import zipfile
from pathlib import Path
import subprocess
import shutil


KAGGLE_DATASET = "pythonafroz/solar-panel-images"
DOWNLOAD_DIR = Path("data/raw")
EXTRACT_DIR = Path("Faulty_solar_panel")


def download_dataset():
    """
    Download dataset from Kaggle using Kaggle CLI.
    Requires kaggle.json to be configured.
    """
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(DOWNLOAD_DIR),
            "--unzip",
        ],
        check=True,
    )

    # Normalize directory structure
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)

    extracted = next(DOWNLOAD_DIR.glob("*"), None)
    if extracted:
        shutil.move(str(extracted), str(EXTRACT_DIR))

    print(f"Dataset ready at: {EXTRACT_DIR.resolve()}")


if __name__ == "__main__":
    download_dataset()
