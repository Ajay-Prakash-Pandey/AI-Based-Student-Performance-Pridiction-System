"""Training entry point for the GitHub repository layout."""

from pathlib import Path
import runpy


TRAIN_PATH = Path(__file__).resolve().parent.parent / "Eda" / "train.py"

runpy.run_path(str(TRAIN_PATH), run_name="__main__")
