"""Utility helpers for model artifact paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def model_path(filename: str) -> Path:
    """Return an absolute path inside the models directory."""
    return MODELS_DIR / filename
