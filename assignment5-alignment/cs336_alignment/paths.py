from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _env_path("CS336_ALIGNMENT_DATA_DIR", PROJECT_ROOT / "data")
MODEL_DIR = _env_path("CS336_ALIGNMENT_MODEL_DIR", DATA_DIR / "a5-alignment" / "models")
OUTPUT_DIR = _env_path("CS336_ALIGNMENT_OUTPUT_DIR", PROJECT_ROOT / "outputs")


def model_path(model_name: str) -> Path:
    return MODEL_DIR / model_name


def train_data_path(data_name: str) -> Path:
    return DATA_DIR / data_name / "train.jsonl"


def test_data_path(data_name: str) -> Path:
    return DATA_DIR / data_name / "test.jsonl"


def prompt_template_path(template_name: str = "r1_zero.prompt") -> Path:
    return Path(__file__).resolve().parent / "prompts" / template_name


def log_path(*parts: str) -> Path:
    return OUTPUT_DIR / "logs" / Path(*parts)


def result_path(*parts: str) -> Path:
    return OUTPUT_DIR / "results" / Path(*parts)


def checkpoint_path(*parts: str) -> Path:
    return OUTPUT_DIR / "models" / Path(*parts)
