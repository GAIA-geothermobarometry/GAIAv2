"""Central inference API for GAIA_v2.

Exposes :func:`run_inference`, which is the single entry point used by both
the Streamlit UI and the test suite. It is fully independent of Streamlit.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from .config import MAX_VALUE, OUTPUT_UNITS, TARGETS
from .models import EnsembleModel
from .preprocessing import apply_chromium_mode, compute_components, feature_matrix
from .validation import validate_chromium_mode, validate_dataframe


@lru_cache(maxsize=None)
def _get_ensemble(target: str, chromium_mode: str) -> EnsembleModel:
    """Cached ensemble loader (process-wide). Streamlit-specific caching
    (``st.cache_resource``) wraps this at the app layer; this cache makes
    sure models are not reloaded within a single Python process/tests."""
    return EnsembleModel(target=target, chromium_mode=chromium_mode)


def preload_models() -> None:
    """Eagerly load every ensemble (both chromium modes, both targets)."""
    for chromium_mode in ("with_chromium", "without_chromium"):
        for target in TARGETS:
            _get_ensemble(target, chromium_mode)


def run_inference(dataframe: pd.DataFrame, chromium_mode: str) -> pd.DataFrame:
    """Run the full GAIA_v2 inference pipeline on raw input data.

    Parameters
    ----------
    dataframe:
        Raw uploaded data containing the required metadata + oxide columns
        (see ``gaia.config.REQUIRED_INPUT_COLUMNS``).
    chromium_mode:
        Either ``"with_chromium"`` (chromium was measured; the model family
        trained with real chromium values is used) or ``"without_chromium"``
        (chromium was not measured; the chromium-related components are
        zeroed exactly as during training of the no-chromium model family).

    Returns
    -------
    pandas.DataFrame
        One row per input sample, preserving input row order, with columns:
        metadata columns, ``predicted_pressure``/``predicted_pressure_std``
        (kbar), ``predicted_temperature``/``predicted_temperature_std`` (°C),
        ``cpx_selection`` (bool quality flag) and ``chromium_mode``.
        Samples failing ``cpx_selection`` get ``NaN`` predictions instead of
        a fabricated value.
    """
    validate_chromium_mode(chromium_mode)
    clean_df = validate_dataframe(dataframe)

    prep = compute_components(clean_df)
    components = apply_chromium_mode(prep["components"], chromium_mode)

    x = feature_matrix(components)
    x_tensor = torch.from_numpy(x)

    results = {}
    for target in TARGETS:
        ensemble = _get_ensemble(target, chromium_mode)
        raw = ensemble.predict_raw(x_tensor).numpy()  # (n_models, n_samples)
        denorm = raw * MAX_VALUE[target]
        results[f"predicted_{target}"] = denorm.mean(axis=0)
        results[f"predicted_{target}_std"] = denorm.std(axis=0)

    checks = prep["checks"]
    out = prep["components"][["Index", "sample", "notes", "notes.1"]].copy()
    valid = checks["cpx_selection"].to_numpy()

    for target in TARGETS:
        mean_col = f"predicted_{target}"
        std_col = f"predicted_{target}_std"
        out[mean_col] = np.where(valid, results[mean_col], np.nan)
        out[std_col] = np.where(valid, results[std_col], np.nan)
        out[f"{mean_col}_unit"] = OUTPUT_UNITS[target]

    out["cpx_selection"] = valid
    out["chromium_mode"] = chromium_mode
    return out

