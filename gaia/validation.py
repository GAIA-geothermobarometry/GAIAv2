"""Input validation for GAIA_v2.

Validates raw uploaded tabular data BEFORE any chemical preprocessing is
attempted. Designed to give clear, scientifically meaningful error messages
instead of raw Python tracebacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CHROMIUM_MODES, METADATA_COLUMNS, OXIDE_COLUMNS, REQUIRED_INPUT_COLUMNS, TOTAL_COLUMN


class ValidationError(Exception):
    """Raised when the uploaded input data cannot be safely processed."""


NUMERIC_COLUMNS = OXIDE_COLUMNS + [TOTAL_COLUMN]


def validate_chromium_mode(chromium_mode: str) -> None:
    if chromium_mode not in CHROMIUM_MODES:
        raise ValidationError(
            f"Modalità cromo non valida: '{chromium_mode}'. Valori ammessi: {CHROMIUM_MODES}."
        )


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw input DataFrame and return a cleaned copy with columns
    reordered to the canonical ``REQUIRED_INPUT_COLUMNS`` order.

    Raises :class:`ValidationError` with a clear, user-facing message on any
    problem. Never silently fixes/reorders/replaces data beyond column
    ordering (which does not change values) and never replaces invalid
    values with zero.
    """
    if df is None or df.empty:
        raise ValidationError("Il file caricato è vuoto o non contiene righe di dati.")

    # duplicated columns
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        raise ValidationError(f"Colonne duplicate nel file: {sorted(set(dup_cols))}.")

    # missing required columns
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(
            "Colonne obbligatorie mancanti: " + ", ".join(missing) +
            ". Le colonne richieste sono, nell'ordine: " + ", ".join(REQUIRED_INPUT_COLUMNS)
        )

    # unsupported / unexpected columns are allowed to exist (e.g. extra
    # metadata) but must not shadow required names; here we just keep the
    # required subset in canonical order without dropping the rest silently.
    extra = [c for c in df.columns if c not in REQUIRED_INPUT_COLUMNS]

    out = df[REQUIRED_INPUT_COLUMNS + extra].copy()

    # numeric checks on oxide + total columns
    numeric_part = out[NUMERIC_COLUMNS]
    non_numeric_mask = numeric_part.apply(lambda col: pd.to_numeric(col, errors="coerce")).isna() & numeric_part.notna()
    if non_numeric_mask.to_numpy().any():
        bad_cols = numeric_part.columns[non_numeric_mask.any(axis=0)].tolist()
        raise ValidationError(
            f"Valori non numerici trovati nelle colonne: {bad_cols}. "
            "Inserire '0' o lasciare la cella vuota se l'ossido non è stato analizzato."
        )

    numeric_converted = numeric_part.apply(pd.to_numeric, errors="coerce")

    # missing values: legacy convention treats blank cells as "not analysed"
    # -> allowed, but must be filled with 0 explicitly (never silently)
    if numeric_converted.isna().to_numpy().any():
        # Not an error: legacy app treats blanks as 0 (not analysed). We make
        # this transformation explicit and documented, applied only here.
        numeric_converted = numeric_converted.fillna(0.0)

    if np.isinf(numeric_converted.to_numpy()).any():
        raise ValidationError(
            "Valori infiniti trovati nei dati numerici. Controllare il file di input."
        )

    if (numeric_converted < 0).to_numpy().any():
        bad_cols = numeric_converted.columns[(numeric_converted < 0).any(axis=0)].tolist()
        raise ValidationError(
            f"Valori negativi non ammessi nelle colonne: {bad_cols}."
        )

    out[NUMERIC_COLUMNS] = numeric_converted

    # metadata columns: fill missing identifiers so downstream code never
    # crashes on NaN sample IDs (does not affect scientific computation)
    for c in METADATA_COLUMNS:
        if out[c].isna().any():
            out[c] = out[c].astype(object).where(out[c].notna(), "")

    return out

