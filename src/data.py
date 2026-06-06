"""Data loading + column-group helpers.

Thin, importable, side-effect-free. Notebooks import these so EDA stays a
display layer over a single canonical loader (no per-notebook CSV parsing).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import (
    DICT_INTERVENABLE,
    ID_COLS,
    OUTCOME_COLS,
    PATHS,
    PRIOR_UNDERWRITER_COLS,
)

# Parsed once; cheap.
_TS_COL = "application_timestamp"


# --------------------------------------------------------------------------- #
# Raw loaders
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _read(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    if _TS_COL in df.columns:
        df[_TS_COL] = pd.to_datetime(df[_TS_COL], errors="coerce")
    return df


def load_train() -> pd.DataFrame:
    return _read(str(PATHS.train)).copy()


def load_validation() -> pd.DataFrame:
    return _read(str(PATHS.validation)).copy()


def load_test() -> pd.DataFrame:
    return _read(str(PATHS.test)).copy()


def load_all() -> dict[str, pd.DataFrame]:
    return {"train": load_train(), "validation": load_validation(), "test": load_test()}


def load_dictionary() -> pd.DataFrame:
    return pd.read_csv(PATHS.data_dictionary)


def load_cohort_defs() -> pd.DataFrame:
    df = pd.read_csv(PATHS.cohort_defs, parse_dates=["start_date", "end_date"])
    return df


def load_intervention_queries() -> pd.DataFrame:
    return pd.read_csv(PATHS.intervention_queries)


# --------------------------------------------------------------------------- #
# Cohort assignment (Deliverable B)
# --------------------------------------------------------------------------- #
def assign_cohort_week(df: pd.DataFrame, cohort_defs: pd.DataFrame | None = None) -> pd.Series:
    """Map application_timestamp -> cohort_week (1..13) via the date ranges.

    Returns an Int64 series (nullable); dates outside all ranges -> <NA>.
    """
    if cohort_defs is None:
        cohort_defs = load_cohort_defs()
    ts = pd.to_datetime(df[_TS_COL], errors="coerce")
    day = ts.dt.normalize()
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for _, row in cohort_defs.iterrows():
        m = (day >= row["start_date"]) & (day <= row["end_date"])
        out[m] = int(row["cohort_week"])
    return out


# --------------------------------------------------------------------------- #
# Column-group helpers (derived from the data dictionary)
# --------------------------------------------------------------------------- #
def column_groups() -> dict[str, list[str]]:
    """{group_name: [fields]} straight from data_dictionary.csv."""
    dd = load_dictionary()
    return {g: sorted(s["field"].tolist()) for g, s in dd.groupby("group")}


def feature_columns(include_prior_underwriter: bool = True) -> list[str]:
    """Candidate model features = all columns minus ids and outcome (label) cols.

    prior_underwriter_* are kept by default (re-underwriting framing); pass
    include_prior_underwriter=False to hard-exclude them.
    """
    dd = load_dictionary()
    drop = set(ID_COLS) | set(OUTCOME_COLS)
    if not include_prior_underwriter:
        drop |= set(PRIOR_UNDERWRITER_COLS)
    return [c for c in dd["field"].tolist() if c not in drop]


def labeled_mask(df: pd.DataFrame) -> pd.Series:
    """True where a repayment outcome is observed (prior-approved + matured)."""
    return df["default_flag"].notna()


def intervenable_fields(source: str = "dict") -> list[str]:
    """source='dict' -> data-dictionary flag; 'queries' -> actually queried in C."""
    if source == "dict":
        return list(DICT_INTERVENABLE)
    if source == "queries":
        return sorted(load_intervention_queries()["feature_name"].unique())
    raise ValueError(source)


def numeric_feature_columns(df: pd.DataFrame, **kw) -> list[str]:
    cols = feature_columns(**kw)
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def categorical_feature_columns(df: pd.DataFrame, **kw) -> list[str]:
    """Dict-declared categoricals (anonymized integer codes) present in df."""
    dd = load_dictionary()
    cats = set(dd.loc[dd["dtype"] == "categorical", "field"])
    cols = feature_columns(**kw)
    return [c for c in cols if c in cats and c in df.columns]
