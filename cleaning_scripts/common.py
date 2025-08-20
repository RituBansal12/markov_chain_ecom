import os
import json
import logging
from pathlib import Path


def project_root() -> Path:
    # scripts/ directory is under project root
    return Path(__file__).resolve().parents[1]


def get_paths():
    root = project_root()
    data = root / "data"
    clean = root / "data_clean"
    reports = root / "reports"
    return root, data, clean, reports


def ensure_dirs():
    _, _, clean, reports = get_paths()
    clean.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)


DEFAULT_CHUNK_ROWS = int(os.environ.get("CHUNK_ROWS", 1_000_000))


def configure_logging(name: str = "pipeline"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
