"""Chemical preprocessing: clinopyroxene major-element oxides (wt%) ->
canonical 11-component feature vector used by the PyTorch models.

This is a direct, behavior-preserving port of the formula implemented in
``GAIA_legacy/GAIA-main/functions/preprocessing.py`` (function
``preprocessing``). The chemistry (cation calculation on a 4-oxygen basis,
T/M site partitioning, component computation) is NOT reinterpreted; only the
code has been reorganized to operate on explicitly named columns instead of
positional slicing, and to separate "compute components" from "compute
validation checks".

Both the "with_chromium" and "without_chromium" inference flows call this
SAME function. The only difference between the two flows is applied
afterwards, in :func:`apply_chromium_mode`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CHROMIUM_FEATURE_COLUMNS,
    CHROMIUM_MODE_WITHOUT,
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    MOLECULAR_WEIGHTS,
    MOLES_OF_OXIDE,
    OXIDE_COLUMNS,
)

_COMPONENT_NAMES = FEATURE_COLUMNS  # alias for readability, exact same order


def compute_components(df: pd.DataFrame) -> dict:
    """Compute clinopyroxene structural components from major-element oxides.

    Parameters
    ----------
    df:
        DataFrame containing (at least) the columns
        ``METADATA_COLUMNS + OXIDE_COLUMNS + ['total']`` in that exact order.
        Validation of this precondition is the responsibility of
        :mod:`gaia.validation`; this function assumes it already holds.

    Returns
    -------
    dict with keys:
        ``components``      : DataFrame, metadata columns + 11 component columns (FEATURE_COLUMNS order)
        ``sum_of_components``: Series, row-wise sum of the 11 components
        ``cations``          : DataFrame, cations per formula unit (debug/reporting)
        ``site_T``           : DataFrame, tetrahedral site occupancy
        ``site_M``           : DataFrame, M1+M2 site occupancy
        ``classifications``  : DataFrame, Wo/En/Fs/Q/J classification values
        ``major``            : DataFrame, recomputed major-element oxide wt% (with split Fe2O3/FeO)
        ``checks``           : DataFrame, boolean quality checks (see compute_checks)
    """
    df = df.reset_index(drop=True)
    n = len(df)

    mw_arr = np.array([MOLECULAR_WEIGHTS[c if c != "FeO tot" else "FeO"] for c in OXIDE_COLUMNS])
    mole_ox = np.array(MOLES_OF_OXIDE)

    oxides = df[OXIDE_COLUMNS].astype(float)

    # cations per formula unit on a 4-oxygen basis, then normalized (corr_fact)
    cat_raw = oxides / mw_arr * mole_ox
    corr_fact = 4 / cat_raw.sum(axis=1)
    cat = cat_raw.multiply(corr_fact, axis=0)

    # Fe3+ / Fe2+ partitioning by charge balance
    charges_of_column = np.array([4, 4, 3, 3, 2, 2, 2, 2, 2, 1, 1])
    charge = (cat * charges_of_column).sum(axis=1)
    difference = 12 - charge

    fe_tot = cat["FeO tot"]
    fe3 = pd.Series(0.0, index=cat.index)
    fe2 = pd.Series(0.0, index=cat.index)

    mask_low = fe_tot < difference
    fe3[mask_low] = fe_tot[mask_low]
    fe2[mask_low] = 0.0
    mask_high = ~mask_low
    fe3[mask_high] = difference[mask_high]
    fe2[mask_high] = (fe_tot - difference)[mask_high]
    mask_neg = difference < 0
    fe3[mask_neg] = 0.0
    fe2[mask_neg] = fe_tot[mask_neg]

    # major element oxide wt% recomputed with split Fe2O3/FeO (for reporting only)
    fe2o3_wt = fe3 / corr_fact * MOLECULAR_WEIGHTS["Fe2O3"] / 2
    feo_wt = fe2 / corr_fact * MOLECULAR_WEIGHTS["FeO"]
    major = df[METADATA_COLUMNS].copy()
    for c in OXIDE_COLUMNS:
        if c != "FeO tot":
            major[c] = df[c]
    major["Fe2O3"] = fe2o3_wt
    major["FeO"] = feo_wt
    non_meta_major_cols = [c for c in major.columns if c not in METADATA_COLUMNS]
    major["tot"] = major[non_meta_major_cols].sum(axis=1)

    # cation dataframe with renamed columns and Fe split into Fe3/Fe2
    cat_named = cat.rename(columns={
        "SiO2": "Si", "TiO2": "Ti", "Al2O3": "Al", "Cr2O3": "Cr", "MnO": "Mn",
        "NiO": "Ni", "MgO": "Mg", "CaO": "Ca", "Na2O": "Na", "K2O": "K",
    }).drop(columns=["FeO tot"])
    cat_named["Fe3"] = fe3
    cat_named["Fe2"] = fe2

    charges_of_column_new = np.array([4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1])
    charge_cols = ["Si", "Ti", "Al", "Cr", "Fe3", "Fe2", "Mn", "Ni", "Mg", "Ca", "Na", "K"]
    charge_balanced = (cat_named[charge_cols] * charges_of_column_new).sum(axis=1)

    # --- T / M site partitioning ---
    T = pd.DataFrame(index=cat_named.index, columns=["Si", "Al", "Ti", "Fe3"], dtype=float)
    T["Si"] = cat_named["Si"]

    al = pd.Series(0.0, index=T.index)
    m = T["Si"] + cat_named["Al"] >= 2
    al[m] = 2 - T["Si"][m]
    m2 = T["Si"] >= 2
    al[m2] = 0.0
    m3 = T["Si"] + cat_named["Al"] < 2
    al[m3] = cat_named["Al"][m3]
    T["Al"] = al

    ti = pd.Series(0.0, index=T.index)
    m = T["Si"] + T["Al"] + cat_named["Ti"] >= 2
    ti[m] = 2 - (T["Si"] + T["Al"])[m]
    m2 = T["Si"] + T["Al"] >= 2
    ti[m2] = 0.0
    m3 = T["Si"] + T["Al"] + cat_named["Ti"] < 2
    ti[m3] = cat_named["Ti"][m3]
    T["Ti"] = ti

    fe3_t = pd.Series(0.0, index=T.index)
    m = T["Si"] + T["Al"] + T["Ti"] + cat_named["Fe3"] >= 2
    fe3_t[m] = (2 - (T["Si"] + T["Al"] + T["Ti"]))[m]
    m3 = T["Si"] + T["Al"] + T["Ti"] + cat_named["Fe3"] < 2
    fe3_t[m3] = cat_named["Fe3"][m3]
    m2 = T["Si"] + T["Al"] + T["Ti"] >= 2
    fe3_t[m2] = 0.0
    T["Fe3"] = fe3_t

    m_cols = ["Mg", "Fe2", "Fe3", "Al", "Ti", "Cr", "Ni", "Mn", "Ca", "Na", "K"]
    M = cat_named[m_cols].copy()
    M["Al"] = M["Al"] - T["Al"]
    M["Fe3"] = M["Fe3"] - T["Fe3"]
    M["Ti"] = M["Ti"] - T["Ti"]

    # --- classification (Q/J diagram quantities) ---
    classif = pd.DataFrame(index=cat_named.index, columns=["Fs", "Wo", "En", "Q", "J"], dtype=float)
    den = M[["Fe2", "Fe3", "Mn", "Mg", "Ca"]].sum(axis=1)
    classif["Fs"] = M[["Fe2", "Fe3", "Mn"]].sum(axis=1) / den * 100
    classif["Wo"] = M["Ca"] / den * 100
    classif["En"] = M["Mg"] / den * 100
    classif["Q"] = M[["Fe2", "Ca", "Mg"]].sum(axis=1)
    classif["J"] = M["Na"] * 2

    # --- component computation (mutates working copies of M / T) ---
    Mw = M.fillna(0).copy()
    Tw = T.copy()

    comp = pd.DataFrame(0.0, index=df.index, columns=_COMPONENT_NAMES)

    # 1) CaTiAl2O6
    cond = (Tw["Al"] + Tw["Fe3"]) >= 2 * Mw["Ti"]
    comp["CaTiAl2O6"] = np.where(cond, Mw["Ti"], (Tw["Al"] + Tw["Fe3"]) / 2)

    Tw["Si"] = Tw["Si"] + Tw["Ti"]
    Tw["Al"] = Tw["Al"] + Tw["Fe3"] - 2 * comp["CaTiAl2O6"]
    Mw["Ti"] = Mw["Ti"] - comp["CaTiAl2O6"]
    Mw["Ca"] = Mw["Ca"] - comp["CaTiAl2O6"]

    # 2-5) CaTs, Es, CaCrTs, NaCrSi2O6
    R1 = pd.Series(0.0, index=Mw.index)
    pos = Mw["Al"] > 0
    R1[pos] = Mw["Fe3"][pos] / Mw["Al"][pos]

    cond = Mw["Al"] > Tw["Al"] / (R1 + 1)
    comp["CaTs"] = np.where(cond, Tw["Al"] / (R1 + 1), Mw["Al"])

    cond = Mw["Fe3"] > (Tw["Al"] - comp["CaTs"])
    comp["Es"] = np.where(cond, Tw["Al"] - comp["CaTs"], Mw["Fe3"])

    cond = Mw["Cr"] > (Tw["Al"] - (comp["Es"] + comp["CaTs"]))
    comp["CaCrTs"] = np.where(cond, Tw["Al"] - comp["Es"] - comp["CaTs"], Mw["Cr"])

    cond = (Mw["Cr"] - comp["CaCrTs"]) > Mw["Na"]
    comp["NaCrSi2O6"] = np.where(cond, Mw["Na"], Mw["Cr"] - comp["CaCrTs"])

    Tw["Si"] = Tw["Si"] - (comp["CaTs"] + comp["Es"] + comp["CaCrTs"] + 2 * comp["NaCrSi2O6"])
    Tw["Al"] = Tw["Al"] - (comp["CaTs"] + comp["Es"] + comp["CaCrTs"])
    Mw["Fe3"] = Mw["Fe3"] - comp["Es"]
    Mw["Al"] = Mw["Al"] - comp["CaTs"]
    Mw["Cr"] = Mw["Cr"] - comp["CaCrTs"] - comp["NaCrSi2O6"]
    Mw["Ca"] = Mw["Ca"] - comp["CaTs"] - comp["Es"] - comp["CaCrTs"]
    Mw["Na"] = Mw["Na"] - comp["NaCrSi2O6"]

    # 6-7) Jd, Ae
    cond = Mw["Al"] > Mw["Na"] / (R1 + 1)
    comp["Jd"] = np.where(cond, Mw["Na"] / (R1 + 1), Mw["Al"])

    cond = Mw["Fe3"] > (Mw["Na"] - comp["Jd"])
    comp["Ae"] = np.where(cond, Mw["Na"] - comp["Jd"], Mw["Fe3"])

    Tw["Si"] = Tw["Si"] - 2 * (comp["Ae"] + comp["Jd"])
    Mw["Fe3"] = Mw["Fe3"] - comp["Ae"]
    Mw["Al"] = Mw["Al"] - comp["Jd"]
    Mw["Na"] = Mw["Na"] - (comp["Ae"] + comp["Jd"])

    # 8-9) Di, Hd
    R2 = pd.Series(0.0, index=Mw.index)
    pos = Mw["Mg"] > 0
    R2[pos] = Mw["Fe2"][pos] / Mw["Mg"][pos]

    cond = Mw["Mg"] > Mw["Ca"] / (R2 + 1)
    comp["Di"] = np.where(cond, Mw["Ca"] / (R2 + 1), Mw["Mg"])
    comp.loc[Mw["Ca"] <= 0, "Di"] = 0.0

    cond = Mw["Fe2"] > (Mw["Ca"] - comp["Di"])
    comp["Hd"] = np.where(cond, Mw["Ca"] - comp["Di"], Mw["Fe2"])
    comp.loc[Mw["Ca"] <= 0, "Hd"] = 0.0

    Tw["Si"] = Tw["Si"] - 2 * (comp["Di"] + comp["Hd"])
    Mw["Mg"] = Mw["Mg"] - comp["Di"]
    Mw["Fe2"] = Mw["Fe2"] - comp["Hd"]
    Mw["Ca"] = Mw["Ca"] - comp["Hd"] - comp["Di"]

    # 10-11) En(Mg+Ni), Fs(Fe+Mn)
    comp["En(Mg+Ni)"] = (Mw["Mg"] + Mw["Ni"]) / 2
    comp["Fs(Fe+Mn)"] = (Mw["Fe2"] + Mw["Mn"]) / 2

    sum_comp = comp.sum(axis=1).astype(float).round(3)
    comp_rounded = comp.astype("float64").round(3)

    checks = _compute_checks(classif, major, sum_comp, T, M, charge_balanced, comp)

    components_out = pd.concat([df[METADATA_COLUMNS].reset_index(drop=True), comp_rounded], axis=1)

    return {
        "components": components_out,
        "sum_of_components": sum_comp,
        "cations": cat_named,
        "site_T": T.astype("float64").round(3),
        "site_M": M.astype("float64").round(3),
        "classifications": classif.round(2),
        "major": major.round(2),
        "checks": checks,
    }


def _compute_checks(classif, major, sum_comp, site_T, site_M, charge_balanced, comp) -> pd.DataFrame:
    """Reproduce the exact quality-control checks from the legacy app.

    A sample failing ``cpx_selection`` is not a reliable clinopyroxene
    analysis and its predictions should be flagged as not computable.
    """
    ck = pd.DataFrame(index=classif.index)
    ck["Wo"] = (classif["Wo"] > 20) & (classif["Wo"] < 55)
    ck["J"] = classif["J"] < 1
    ck["Fs"] = (classif["Fs"] > 5) & (classif["Fs"] < 50)
    ck["Wt%"] = (major["tot"] > 97.5) & (major["tot"] < 102.5)
    ck["components"] = (sum_comp > 0.95) & (sum_comp < 1.05)
    ck["Si apfu"] = site_T["Si"] <= 2
    ck["CaTiAl2O6"] = comp["CaTiAl2O6"] >= 0
    ck["T_site"] = (site_T.sum(axis=1) > 1.95) & (site_T.sum(axis=1) < 2.05)
    ck["M_site"] = (site_M.sum(axis=1) > 1.95) & (site_M.sum(axis=1) < 2.05)
    ck["charge"] = (charge_balanced > 11.95) & (charge_balanced < 12.05)
    ck["cpx_selection"] = ck.all(axis=1)
    return ck


def apply_chromium_mode(components: pd.DataFrame, chromium_mode: str) -> pd.DataFrame:
    """Apply the chromium mode to an already-computed component DataFrame.

    All three flows share :func:`compute_components`. This function is the
    ONLY place where chromium-related features are (unconditionally) zeroed,
    and it must be called AFTER chemical preprocessing and BEFORE model
    inference, exactly reproducing the training-time procedure::

        if not Use_chrome:
            df['CaCrTs'] = 0
            df['NaCrSi2O6'] = 0

    ``CHROMIUM_MODE_MIXED`` ("mixed_chromium", the model family trained on
    both measured chromium and NaN-chromium-replaced-with-zero, see
    ``dan_003_Crnan_to_zero`` in the original training workspace) requires
    NO additional zeroing here: measured chromium must be preserved as-is,
    and missing chromium is already mapped to 0 upstream, in
    ``gaia.validation.validate_dataframe`` (blank/NaN oxide cells -> 0.0,
    the same general convention used for every oxide column). Feeding
    Cr2O3 = 0 into :func:`compute_components` yields CaCrTs = NaCrSi2O6 = 0
    exactly, numerically identical to the training-time
    ``fillna(0)`` applied to the derived component columns. Therefore
    ``CHROMIUM_MODE_MIXED`` behaves exactly like ``CHROMIUM_MODE_WITH`` at
    this stage: the two flows differ only in which trained ensemble
    (checkpoint set) is used for inference (see ``gaia.config.CHROMIUM_MODES``
    / ``gaia.models.EnsembleModel``).

    Parameters
    ----------
    components:
        DataFrame containing at least the ``FEATURE_COLUMNS``.
    chromium_mode:
        One of ``config.CHROMIUM_MODE_WITH`` / ``config.CHROMIUM_MODE_WITHOUT``
        / ``config.CHROMIUM_MODE_MIXED``.
    """
    out = components.copy()
    if chromium_mode == CHROMIUM_MODE_WITHOUT:
        for col in CHROMIUM_FEATURE_COLUMNS:
            out[col] = 0.0
    return out


def feature_matrix(components: pd.DataFrame) -> np.ndarray:
    """Extract the canonical (N, 11) float32 feature matrix, in the exact
    FEATURE_COLUMNS order, ready to be fed to the PyTorch models."""
    return components[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

