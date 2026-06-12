import logging
import json
import time
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR    = ROOT_DIR / "src"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

LABELS        = ["high_risk", "medium_risk", "low_risk"]


def get_logger(name):

    logger = logging.getLogger(name)

    if not logger.handlers:

        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s — %(message)s"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    return logger

def save_json(obj: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self.start_time

    def __str__(self):
        return f"{self.elapsed:.2f}sec"