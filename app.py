"""GAIA_v2 - Streamlit app.

Thin presentation layer only. All scientific logic lives in the `gaia`
package and is fully testable independently of Streamlit.
"""
from __future__ import annotations

import traceback
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from gaia.config import (
    CHROMIUM_MODE_WITH,
    CHROMIUM_MODE_WITHOUT,
    EXAMPLES_DIR,
    OUTPUT_UNITS,
    REQUIRED_INPUT_COLUMNS,
)
from gaia.inference import preload_models, run_inference
from gaia.validation import ValidationError

LOGO_PATH = Path(__file__).resolve().parent / "logo_gaiav2.png"
_logo_image = Image.open(LOGO_PATH) if LOGO_PATH.exists() else None

st.set_page_config(
    page_title="GAIA v2 - Geothermobarometry",
    page_icon=_logo_image if _logo_image is not None else "🌋",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading PyTorch models...")
def _load_models_once() -> bool:
    preload_models()
    return True


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return buffer.getvalue()


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xls") or name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValidationError(
        "Unsupported file format. Please upload a .csv, .xls or .xlsx file."
    )


def _plot_histograms(results: pd.DataFrame) -> None:
    """Plot P and T prediction histograms for valid (cpx_selection) samples,
    reproducing the legacy `plothist` behaviour."""
    valid = results.loc[results["cpx_selection"]]
    targets = [
        ("predicted_pressure", "P distribution", "P (kbar)", "tab:green"),
        ("predicted_temperature", "T distribution", "T (°C)", "tab:red"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (col, title, xlabel, color) in zip(axes, targets):
        data = valid[col].dropna()
        if len(data) > 0:
            ax.hist(data.values, density=True, edgecolor="k", color=color)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel, fontsize=13)
    fig.tight_layout(pad=2.0)
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_logo = st.columns([3, 1])
with col_title:
    st.title("GAIA v2")
    st.subheader("Geo Artificial Intelligence thermobArometry")
    st.write(
        "PyTorch deep learning ensemble for Pressure (kbar) and Temperature (°C) "
        "estimates of volcano plumbing systems, based on clinopyroxene composition."
    )
    st.caption(
        "Standalone version (GAIA_v2), based on the final PyTorch models. "
        "See README.md for scientific details, limitations and attribution."
    )
with col_logo:
    if _logo_image is not None:
        st.image(_logo_image, width=500)

with st.expander("Instructions and input file format", expanded=False):
    st.markdown(
        "The uploaded file (csv, xls or xlsx) must contain the following columns, "
        "in the indicated order:\n\n"
        f"`{', '.join(REQUIRED_INPUT_COLUMNS)}`\n\n"
        "- ***Index***, ***sample***, ***notes***, ***notes.1***: identify the sample "
        "(any text).\n"
        "- Major element oxides (wt%): ***SiO2, TiO2, Al2O3, Cr2O3, FeO tot, MnO, NiO, MgO, CaO, "
        "Na2O, K2O***. If an oxide has not been analysed or is below detection limit, "
        "enter 0 or leave the cell empty.\n"
        "- ***tot***: analytical total (informational, not used in the calculation)."
    )
    example_path = EXAMPLES_DIR / "input_example.xlsx"
    if example_path.exists():
        with open(example_path, "rb") as f:
            st.download_button(
                "Download an example file",
                data=f.read(),
                file_name="input_example.xlsx",
            )

# ---------------------------------------------------------------------------
# Chromium mode (explicit user choice - never inferred)
# ---------------------------------------------------------------------------
st.header("1. Chromium mode")
chromium_choice = st.radio(
    "Was Chromium (Cr2O3) measured in the uploaded analyses?",
    options=["Yes, Chromium was measured", "No, Chromium was not measured"],
    index=0,
)
chromium_mode = CHROMIUM_MODE_WITH if chromium_choice.startswith("Yes") else CHROMIUM_MODE_WITHOUT

if chromium_mode == CHROMIUM_MODE_WITHOUT:
    st.info(
        "The model family trained by zeroing the chromium-dependent components "
        "(CaCrTs, NaCrSi2O6) will be used, exactly as during training."
    )

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.header("2. Data upload")
uploaded_file = st.file_uploader("Upload a file (.csv, .xls, .xlsx)", type=["csv", "xls", "xlsx"])

raw_df: pd.DataFrame | None = None
if uploaded_file is not None:
    try:
        raw_df = _read_uploaded_file(uploaded_file)
        st.write("Preview of the uploaded data:")
        st.dataframe(raw_df.head(20))
    except Exception:
        st.error(
            "Could not read the uploaded file. Please check that it is a valid, "
            "non-empty csv/xls/xlsx file."
        )
        raw_df = None

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
st.header("3. Prediction")

if raw_df is not None and st.button("Run pre-processing and prediction", type="primary"):
    try:
        _load_models_once()
        results = run_inference(raw_df, chromium_mode=chromium_mode)
    except ValidationError as e:
        st.error(f"Error in the input data: {e}")
    except Exception:
        st.error(
            "An unexpected error occurred while processing the data. "
            "Please check the file format and try again."
        )
        with st.expander("Technical details (for developers)"):
            st.code(traceback.format_exc())
    else:
        n_invalid = (~results["cpx_selection"]).sum()
        if n_invalid:
            st.warning(
                f"{n_invalid} out of {len(results)} sample(s) do not pass the chemical "
                "quality checks (cpx_selection) and are marked as not computable (NaN)."
            )

        st.write("Predictions (one row per sample, same order as the input):")
        display_cols = [
            "Index", "sample", "notes", "notes.1",
            "predicted_pressure", "predicted_pressure_std",
            "predicted_temperature", "predicted_temperature_std",
            "cpx_selection",
        ]
        st.dataframe(results[display_cols])
        st.caption(
            f"Units: Pressure in {OUTPUT_UNITS['pressure']}, "
            f"Temperature in {OUTPUT_UNITS['temperature']}. "
            "The reported uncertainty is the standard deviation across the ensemble models."
        )

        st.subheader("Prediction histograms")
        if results["cpx_selection"].any():
            _plot_histograms(results)
        else:
            st.info("No valid sample (cpx_selection) to plot.")

        excel_bytes = _to_excel_bytes(results)
        st.download_button(
            "Download results (xlsx)",
            data=excel_bytes,
            file_name="GAIA_v2_predictions.xlsx",
        )
elif raw_df is None:
    st.info("Upload a file to proceed with the prediction.")

