"""Tests for gaia.preprocessing: shared preprocessing pipeline and chromium
zeroing behavior."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaia.config import (
    CHROMIUM_FEATURE_COLUMNS,
    CHROMIUM_MODE_MIXED,
    CHROMIUM_MODE_WITH,
    CHROMIUM_MODE_WITHOUT,
    FEATURE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
)
from gaia.preprocessing import apply_chromium_mode, compute_components, feature_matrix
from gaia.validation import validate_dataframe


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Index": ["a1", "a2"],
        "sample": ["cpx1", "cpx2"],
        "notes": ["core", "rim"],
        "notes.1": ["erupt I", "erupt I"],
        "SiO2": [52.37, 47.88],
        "TiO2": [0.55, 0.67],
        "Al2O3": [2.89, 3.13],
        "Cr2O3": [0.07, 0.07],
        "FeO tot": [3.89, 7.40],
        "MnO": [0.05, 0.00],
        "NiO": [0.0, 0.01],
        "MgO": [16.13, 14.65],
        "CaO": [23.80, 21.18],
        "Na2O": [0.09, 0.66],
        "K2O": [0.0, 0.03],
        "tot": [99.99, 95.78],
    })


def test_required_columns_present_in_fixture():
    df = _sample_raw_df()
    assert set(REQUIRED_INPUT_COLUMNS).issubset(df.columns)


def test_compute_components_feature_order_deterministic():
    df = _sample_raw_df()
    result = compute_components(df)
    comps = result["components"]
    assert list(comps.columns[-len(FEATURE_COLUMNS):]) == FEATURE_COLUMNS
    # deterministic: running twice gives identical results
    result2 = compute_components(df)
    pd.testing.assert_frame_equal(comps, result2["components"])


def test_compute_components_sum_close_to_one_for_good_analyses():
    df = _sample_raw_df()
    result = compute_components(df)
    # both analyses are realistic cpx compositions -> component sum near 1
    assert (result["sum_of_components"] - 1).abs().max() < 0.05


def test_with_chromium_preserves_chromium_features():
    df = _sample_raw_df()
    comps = compute_components(df)["components"]
    out = apply_chromium_mode(comps, CHROMIUM_MODE_WITH)
    pd.testing.assert_frame_equal(out[CHROMIUM_FEATURE_COLUMNS], comps[CHROMIUM_FEATURE_COLUMNS])


def test_without_chromium_zeroes_only_chromium_features():
    df = _sample_raw_df()
    comps = compute_components(df)["components"]
    out = apply_chromium_mode(comps, CHROMIUM_MODE_WITHOUT)
    assert (out[CHROMIUM_FEATURE_COLUMNS] == 0.0).all().all()
    other_cols = [c for c in FEATURE_COLUMNS if c not in CHROMIUM_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(out[other_cols], comps[other_cols])


def test_feature_matrix_shape_and_dtype():
    df = _sample_raw_df()
    comps = compute_components(df)["components"]
    x = feature_matrix(comps)
    assert x.shape == (2, len(FEATURE_COLUMNS))
    assert x.dtype == np.float32


# ---------------------------------------------------------------------------
# mixed_chromium: all preprocessing goes through validate_dataframe (which
# maps blank/NaN oxide cells, including Cr2O3, to 0.0) + compute_components +
# apply_chromium_mode. mixed_chromium performs NO additional zeroing in
# apply_chromium_mode (same as with_chromium): measured chromium is
# preserved, and missing chromium is already zeroed upstream by
# validate_dataframe, exactly reproducing the dan_003_Crnan_to_zero
# training-time fillna(0) on the derived component columns.
# ---------------------------------------------------------------------------

def _mixed_raw_df() -> pd.DataFrame:
    """3 rows: chromium measured, chromium missing (blank), chromium
    genuinely measured as zero."""
    df = _sample_raw_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.loc[0, "Cr2O3"] = 0.07  # measured, non-zero
    df.loc[1, "Cr2O3"] = None  # missing (blank cell)
    df.loc[2, "Cr2O3"] = 0.0  # measured, genuinely zero
    df["Index"] = ["measured", "missing", "measured_zero"]
    return df


def test_mixed_chromium_all_measured_preserves_values():
    df = _sample_raw_df()  # both rows have a real, non-zero Cr2O3
    comps = compute_components(df)["components"]
    out = apply_chromium_mode(comps, CHROMIUM_MODE_MIXED)
    pd.testing.assert_frame_equal(out[CHROMIUM_FEATURE_COLUMNS], comps[CHROMIUM_FEATURE_COLUMNS])


def test_mixed_chromium_all_missing_zeroes_chromium_features():
    df = _sample_raw_df()
    df["Cr2O3"] = None
    clean = validate_dataframe(df)  # blank Cr2O3 -> 0.0 (upstream, general rule)
    comps = compute_components(clean)["components"]
    out = apply_chromium_mode(comps, CHROMIUM_MODE_MIXED)
    assert (out[CHROMIUM_FEATURE_COLUMNS].abs() < 1e-9).all().all()


def test_mixed_chromium_partial_missing_preserves_measured_and_zeroes_missing():
    df = _mixed_raw_df()
    clean = validate_dataframe(df)
    comps = compute_components(clean)["components"]
    out = apply_chromium_mode(comps, CHROMIUM_MODE_MIXED)

    # row 0 (measured, non-zero Cr) -> non-zero chromium components expected
    assert (out.loc[0, CHROMIUM_FEATURE_COLUMNS].abs() > 0).any()
    # row 1 (missing Cr) -> chromium components zeroed
    assert (out.loc[1, CHROMIUM_FEATURE_COLUMNS].abs() < 1e-9).all()
    # row 2 (measured genuine zero) -> chromium components are (correctly) zero too
    assert (out.loc[2, CHROMIUM_FEATURE_COLUMNS].abs() < 1e-9).all()


def test_mixed_chromium_does_not_alter_non_chromium_features():
    df = _mixed_raw_df()
    clean = validate_dataframe(df)
    comps = compute_components(clean)["components"]
    out = apply_chromium_mode(comps, CHROMIUM_MODE_MIXED)
    other_cols = [c for c in FEATURE_COLUMNS if c not in CHROMIUM_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(out[other_cols], comps[other_cols])


def test_mixed_chromium_row_order_preserved():
    df = _mixed_raw_df()
    clean = validate_dataframe(df)
    comps = compute_components(clean)["components"]
    out = apply_chromium_mode(comps, CHROMIUM_MODE_MIXED)
    assert list(out["Index"]) == ["measured", "missing", "measured_zero"]


