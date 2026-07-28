"""Central configuration for GAIA_v2.

All paths are resolved relative to this file (i.e. relative to the
``GAIA_v2`` package root), so the application works regardless of the
current working directory or the machine it is run on.

This module intentionally uses a simple Python dictionary/constant based
configuration instead of a YAML/JSON configuration framework, since the
number of configurable parameters is small and fixed by the trained model
artifacts (see ``pythorch_models`` in the original training workspace).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to this package -> fully self-contained)
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

# ---------------------------------------------------------------------------
# Required raw input columns (major element oxides, wt%) - identifies a
# single clinopyroxene analysis. Order matches the legacy GAIA app
# (GAIA_legacy/GAIA-main/functions/preprocessing.py).
# ---------------------------------------------------------------------------
METADATA_COLUMNS = ["Index", "sample", "notes", "notes.1"]
OXIDE_COLUMNS = [
    "SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO tot", "MnO", "NiO", "MgO", "CaO", "Na2O", "K2O",
]
# NOTE: the legacy template/example file (GAIA_legacy/GAIA-main/files/input_example.xlsx)
# names this column "tot" (not "total", despite the on-screen documentation
# text in Home.py). It is informational only: it is never used in the
# chemical computation (the oxide totals are recomputed from the 11 oxides).
TOTAL_COLUMN = "tot"
REQUIRED_INPUT_COLUMNS = METADATA_COLUMNS + OXIDE_COLUMNS + [TOTAL_COLUMN]

# Molecular weights (g/mol) used in cation calculation. Order matches
# OXIDE_COLUMNS + Fe2O3 (used internally for Fe3+/Fe2+ splitting).
MOLECULAR_WEIGHTS = {
    "SiO2": 60.084, "TiO2": 79.900, "Al2O3": 101.960, "Cr2O3": 151.990, "FeO": 71.846,
    "MnO": 70.937, "NiO": 74.699, "MgO": 40.304, "CaO": 56.079, "Na2O": 61.979,
    "K2O": 94.196, "Fe2O3": 159.69,
}

# Moles of oxide per cation (used for cation-per-formula-unit normalization).
MOLES_OF_OXIDE = [1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2]

# ---------------------------------------------------------------------------
# Canonical clinopyroxene component feature order (model input, dim = 11).
# This is the EXACT order produced by the chemical preprocessing and
# consumed by the PyTorch models. Do not reorder.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "CaTiAl2O6", "CaTs", "Es", "CaCrTs", "NaCrSi2O6", "Jd", "Ae", "Di", "Hd",
    "En(Mg+Ni)", "Fs(Fe+Mn)",
]
INPUT_DIM = len(FEATURE_COLUMNS)  # 11

# Chromium-bearing components. In the "without_chromium" flow these are
# forced to 0.0 AFTER the chemical preprocessing and BEFORE model inference,
# exactly reproducing the procedure used to train the "No_Chromium" model
# family (see definitive_train_July26.py / load_definitive_models_July26.py
# in the original training workspace: `df['CaCrTs'] = 0; df['NaCrSi2O6'] = 0`).
CHROMIUM_FEATURE_COLUMNS = ["CaCrTs", "NaCrSi2O6"]

# ---------------------------------------------------------------------------
# Chromium modes (explicit, user-selected - never inferred automatically).
#
# CHROMIUM_MODE_MIXED ("mixed_chromium") corresponds to the third model
# family trained in definitive_train_July26.py (experiments == 2:
# Use_chrome=True, set_nan_Cr_to_zero=True; checkpoints copied from
# results_july26/Three_models_run/dan_003_Crnan_to_zero). During training,
# the chromium-dependent component columns ('CaCrTs', 'NaCrSi2O6') were
# NaN for samples where chromium had not been measured (in the
# already-preprocessed training/test datasets, only those two columns were
# NaN, all other components were valid) and were replaced with 0 via
# ``df['CaCrTs'] = df['CaCrTs'].fillna(0)`` (and likewise for
# 'NaCrSi2O6'), while samples with measured chromium kept their real
# (possibly genuinely zero) values untouched.
#
# GAIA_v2 starts from raw oxide wt% (not from precomputed components).
# ``gaia.validation.validate_dataframe`` already fills missing/blank oxide
# cells (including Cr2O3) with 0.0 before chemical preprocessing runs (this
# is the existing, general "blank = not analysed" convention, applied
# identically for all three chromium modes). Feeding Cr2O3 = 0 into
# ``compute_components`` yields CaCrTs = NaCrSi2O6 = 0 exactly (both
# formulas reduce to ``min(..., cation_Cr)`` with cation_Cr = 0), which is
# numerically IDENTICAL to the training-time ``fillna(0)`` applied to the
# derived component columns. Consequently, no extra zeroing step is
# required for CHROMIUM_MODE_MIXED in ``apply_chromium_mode``: measured
# chromium is preserved (real, non-zero cation), missing chromium already
# becomes 0 through the same general validation step used for every other
# oxide. This reproduces the training procedure exactly without
# duplicating preprocessing logic.
# ---------------------------------------------------------------------------
CHROMIUM_MODE_WITH = "with_chromium"
CHROMIUM_MODE_WITHOUT = "without_chromium"
CHROMIUM_MODE_MIXED = "mixed_chromium"
CHROMIUM_MODES = (CHROMIUM_MODE_WITH, CHROMIUM_MODE_WITHOUT, CHROMIUM_MODE_MIXED)

_ARTIFACT_SUBDIR = {
    CHROMIUM_MODE_WITH: "with_chromium",
    CHROMIUM_MODE_WITHOUT: "without_chromium",
    CHROMIUM_MODE_MIXED: "mixed_chromium",
}

# ---------------------------------------------------------------------------
# Targets, ensemble sizes and output de-normalization constants.
# Source of truth: definitive_train_July26.py / load_definitive_models_July26.py
#   max_value = {'temperature': 1400, 'pressure': 10}
#   pressure ensemble size = 100 models, temperature ensemble size = 20 models
# ---------------------------------------------------------------------------
TARGETS = ("pressure", "temperature")

ENSEMBLE_SIZE = {"pressure": 100, "temperature": 20}

# Hidden layer width of the domain-adaptation MLP backbone per target.
HIDDEN_WIDTH = {"pressure": 100, "temperature": 1000}

# Linear de-normalization factor applied to the raw model output to obtain
# physical units (pressure in kbar, temperature in degrees Celsius).
MAX_VALUE = {"pressure": 10.0, "temperature": 1400.0}

OUTPUT_UNITS = {"pressure": "kbar", "temperature": "°C"}


def artifact_dir(chromium_mode: str, target: str) -> Path:
    """Return the directory containing the ensemble checkpoints for a given
    chromium mode and target, relative to the GAIA_v2 project root."""
    if chromium_mode not in CHROMIUM_MODES:
        raise ValueError(f"Unknown chromium_mode '{chromium_mode}', expected one of {CHROMIUM_MODES}")
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}', expected one of {TARGETS}")
    return ARTIFACTS_DIR / _ARTIFACT_SUBDIR[chromium_mode] / target


def checkpoint_paths(chromium_mode: str, target: str) -> list[Path]:
    """Return the sorted list of checkpoint paths for the given ensemble."""
    n = ENSEMBLE_SIZE[target]
    directory = artifact_dir(chromium_mode, target)
    return [directory / f"mod_{i}_.pth" for i in range(n)]

