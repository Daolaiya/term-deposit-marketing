"""Shared helper utilities for the Term Deposit Marketing notebooks.

Both ``notebook_tdm_1.ipynb`` and ``notebook_tdm_2.ipynb`` import from this
module instead of redefining the same helpers locally.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

_BOLD_START = "\033[1m"
_BOLD_END = "\033[0;0m"


def barrier() -> None:
    """Print a horizontal separator between cell outputs."""
    print("\n <<<", "-" * 50, ">>> \n")


def print_bold(text: object) -> None:
    """Print ``text`` using terminal bold escape codes."""
    print(_BOLD_START + str(text) + _BOLD_END)


def bold(text: object) -> str:
    """Return ``text`` wrapped in terminal bold escape codes."""
    return _BOLD_START + str(text) + _BOLD_END


def print_uniques(df: pd.DataFrame) -> None:
    """Print the unique values of every column in ``df``."""
    for feature in df.columns:
        print(bold(feature), "------>", df[feature].unique())


def save_model(model, name: str, folder: str = "models") -> None:
    """Pickle ``model`` to ``<folder>/<name>.pkl``."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(f"{folder}/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model: {name}.pkl saved to {folder}/.")


def load_model(name: str, folder: str = "models"):
    """Load a pickled model from ``<folder>/<name>``."""
    with open(f"{folder}/{name}", "rb") as f:
        return pickle.load(f)


def save_model_dropped(model, name: str) -> None:
    """Pickle ``model`` to ``models_dropped/<name>.pkl``."""
    save_model(model, name, folder="models_dropped")


def load_model_dropped(name: str):
    """Load a pickled model from ``models_dropped/<name>``."""
    return load_model(name, folder="models_dropped")


def na_random_fill(
    series: pd.Series,
    source: Optional[pd.Series] = None,
    random_state: int = 0,
) -> pd.Series:
    """Fill NaN values in ``series`` by sampling from a non-null pool.

    Parameters
    ----------
    series : pd.Series
        Series whose NaN values should be imputed.
    source : pd.Series, optional
        Pool to sample fill values from. Defaults to ``series`` itself when
        imputing a single dataset; pass the *training* series when imputing a
        held-out test set so that fill values do not leak across the split.
    random_state : int, default ``0``
        Seed for reproducibility.

    Returns
    -------
    pd.Series
        A new series with NaN values replaced by random draws from ``source``.
    """
    na_mask = pd.isnull(series)
    n_null = int(na_mask.sum())
    if n_null == 0:
        return series.copy()
    pool = (source if source is not None else series).dropna()
    fill_values = pool.sample(n=n_null, replace=True, random_state=random_state)
    fill_values.index = series.index[na_mask]
    return series.fillna(fill_values)
