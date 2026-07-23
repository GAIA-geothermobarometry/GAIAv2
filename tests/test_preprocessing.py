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
    CHROMIUM_MODE_WITH,
    CHROMIUM_MODE_WITHOUT,
    FEATURE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
)
from gaia.preprocessing import apply_chromium_mode, compute_components, feature_matrix


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

