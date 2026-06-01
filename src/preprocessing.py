"""Shared preprocessing helpers for score normalization and feature mapping."""

import pandas as pd


def score_to_20(value: float, scoring_system: str) -> float:
    """Convert percentage or CGPA scores to the model's 0-20 scale."""
    if pd.isna(value):
        return float("nan")

    scoring_system = str(scoring_system).lower()
    if "10" in scoring_system:
        scaled = value * 2
    elif "4" in scoring_system:
        scaled = value * 5
    elif "20" in scoring_system:
        scaled = value
    else:
        scaled = value / 5

    return float(max(0, min(20, scaled)))


def subject_proxy(area: str) -> str:
    """Map broad Indian subject/program areas to the current model's subject feature."""
    quantitative_keywords = ["math", "science", "engineering", "technology", "medical", "commerce"]
    area_text = str(area).lower()
    return "Mathematics" if any(keyword in area_text for keyword in quantitative_keywords) else "Portuguese"
