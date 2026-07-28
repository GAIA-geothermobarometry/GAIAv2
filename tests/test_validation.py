"""Tests for gaia.validation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaia.config import CHROMIUM_MODE_MIXED, REQUIRED_INPUT_COLUMNS
from gaia.validation import ValidationError, validate_chromium_mode, validate_dataframe


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Index": ["a1"], "sample": ["cpx1"], "notes": ["core"], "notes.1": ["e1"],
        "SiO2": [52.0], "TiO2": [0.5], "Al2O3": [2.9], "Cr2O3": [0.1],
        "FeO tot": [3.9], "MnO": [0.05], "NiO": [0.0], "MgO": [16.0],
        "CaO": [23.8], "Na2O": [0.1], "K2O": [0.0], "tot": [99.9],
    })


def test_valid_input_accepted():
    df = validate_dataframe(_valid_df())
    assert list(df.columns[: len(REQUIRED_INPUT_COLUMNS)]) == REQUIRED_INPUT_COLUMNS


def test_missing_required_column_raises():
    df = _valid_df().drop(columns=["SiO2"])
    with pytest.raises(ValidationError):
        validate_dataframe(df)


def test_empty_dataframe_raises():
    with pytest.raises(ValidationError):
        validate_dataframe(pd.DataFrame())


def test_duplicated_columns_raise():
    df = _valid_df()
    df = pd.concat([df, df[["SiO2"]]], axis=1)
    with pytest.raises(ValidationError):
        validate_dataframe(df)


def test_non_numeric_value_raises():
    df = _valid_df()
    df.loc[0, "SiO2"] = "not-a-number"
    with pytest.raises(ValidationError):
        validate_dataframe(df)


def test_infinite_value_raises():
    df = _valid_df()
    df.loc[0, "SiO2"] = float("inf")
    with pytest.raises(ValidationError):
        validate_dataframe(df)


def test_negative_value_raises():
    df = _valid_df()
    df.loc[0, "SiO2"] = -1.0
    with pytest.raises(ValidationError):
        validate_dataframe(df)


def test_blank_oxide_filled_with_zero_not_error():
    df = _valid_df()
    df.loc[0, "NiO"] = None
    out = validate_dataframe(df)
    assert out.loc[0, "NiO"] == 0.0


def test_invalid_chromium_mode_raises():
    with pytest.raises(ValidationError):
        validate_chromium_mode("maybe_chromium")


def test_mixed_chromium_mode_is_valid():
    validate_chromium_mode(CHROMIUM_MODE_MIXED)  # must not raise


def test_missing_chromium_column_filled_with_zero_for_mixed_mode():
    """Blank Cr2O3 cell -> 0.0 (upstream general convention), while a
    genuinely measured Cr2O3 value in another row is left untouched."""
    df = pd.concat([_valid_df(), _valid_df()], ignore_index=True)
    df.loc[1, "Cr2O3"] = None
    out = validate_dataframe(df)
    assert out.loc[0, "Cr2O3"] == 0.1  # measured value preserved
    assert out.loc[1, "Cr2O3"] == 0.0  # missing -> zero


