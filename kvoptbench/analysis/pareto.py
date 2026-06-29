"""Pareto analysis placeholder for later milestones."""

from __future__ import annotations

import pandas as pd


def quality_adjusted_rps(frame: pd.DataFrame) -> pd.Series:
    if "requests_per_second_mean" not in frame or "quality_score_mean" not in frame:
        return pd.Series(dtype=float)
    return frame["requests_per_second_mean"].fillna(0) * frame["quality_score_mean"].fillna(0)

