"""Reproducible dataset preparation and statistical models for ACDT Project 1.

All displayed statistical effects are *associations*, not causal estimates.
The data and the preprocessing/model formulas follow the supplied notebook.
"""
from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GENRES = ("Action", "Comedy", "Drama", "Horror")
REQUIRED = (
    "title", "metascore", "rating", "votes", "budget", "gross_worldwide",
    "gross_us_canada", "release_date", "genres", "duration",
)


def split_genres(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(g).strip() for g in parsed]
    except (ValueError, SyntaxError):
        pass
    return [x.strip() for x in str(value).split(",")]


def choose_primary(value: str) -> str | None:
    """First listed genre matching one of the four target genres, per notebook."""
    return next((genre for genre in split_genres(value) if genre in GENRES), None)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA / "movies_2022_2025_usd.csv")


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required movie data columns: {missing}")

    df = raw.copy()
    for col in ("metascore", "rating", "votes", "budget", "gross_worldwide", "gross_us_canada"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["primary_genre"] = df["genres"].map(choose_primary)
    df["target_genre_count"] = df["genres"].map(
        lambda v: sum(g in split_genres(v) for g in GENRES)
    )
    valid = (
        df["release_year"].between(2022, 2025)
        & df["metascore"].between(0, 100)
        & (df["budget"] >= 1_000)
        & (df["gross_worldwide"] > 0)
        & (df["votes"] >= 100)
        & df[list(REQUIRED)].notna().all(axis=1)
    )
    validated = df.loc[valid].copy()
    study = validated.dropna(subset=["primary_genre"]).copy()
    study["log_revenue"] = np.log(study["gross_worldwide"])
    study["log_budget"] = np.log(study["budget"])
    return study, validated


def fit_regressions(df: pd.DataFrame):
    simple = smf.ols("log_revenue ~ metascore", data=df).fit(cov_type="HC3")
    multiple = smf.ols(
        "log_revenue ~ metascore + log_budget + C(release_year) + C(primary_genre)", data=df
    ).fit(cov_type="HC3")
    interaction = smf.ols(
        "log_revenue ~ metascore * C(primary_genre) + log_budget + C(release_year)", data=df
    ).fit(cov_type="HC3")
    return simple, multiple, interaction


def per_ten_points(beta: float) -> float:
    return 100 * np.expm1(10 * float(beta))


def prediction(interaction, metascore: int, budget_m: float, genre: str, year: int) -> dict:
    """Exponentiated conditional mean of log revenue with its *mean* CI.

    Not a prediction interval for an individual movie and not a causal effect.
    """
    if genre not in GENRES or not 0 <= metascore <= 100 or budget_m <= 0 or year not in (2022, 2023, 2024, 2025):
        raise ValueError("Invalid prediction input")
    new = pd.DataFrame({
        "metascore": [metascore], "log_budget": [np.log(budget_m * 1_000_000)],
        "release_year": [year], "primary_genre": [genre],
    })
    out = interaction.get_prediction(new).summary_frame(alpha=.05).iloc[0]
    return {
        "usd": float(np.exp(out["mean"])),
        "lo": float(np.exp(out["mean_ci_lower"])),
        "hi": float(np.exp(out["mean_ci_upper"])),
    }


def prediction_curve(interaction, genre: str, budget_m: float, year: int) -> pd.DataFrame:
    xs = np.arange(0, 101, 2)
    new = pd.DataFrame({
        "metascore": xs,
        "log_budget": np.repeat(np.log(budget_m * 1_000_000), len(xs)),
        "release_year": np.repeat(year, len(xs)),
        "primary_genre": np.repeat(genre, len(xs)),
    })
    sf = interaction.get_prediction(new).summary_frame(alpha=.05)
    return pd.DataFrame({
        "metascore": xs, "usd": np.exp(sf["mean"].to_numpy()),
        "lo": np.exp(sf["mean_ci_lower"].to_numpy()),
        "hi": np.exp(sf["mean_ci_upper"].to_numpy()),
    })
