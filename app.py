"""Streamlit entry point for the Student Performance Prediction AI app."""

from pathlib import Path
import runpy


APP_PATH = Path(__file__).resolve().parent / "Eda" / "app.py"

runpy.run_path(str(APP_PATH), run_name="__main__")
