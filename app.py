"""GAIA_v2 - Streamlit app.

Thin presentation layer only. All scientific logic lives in the `gaia`
package and is fully testable independently of Streamlit.
"""
from __future__ import annotations

import traceback
from io import BytesIO

import pandas as pd
import streamlit as st

from gaia.config import (
    CHROMIUM_MODE_WITH,
    CHROMIUM_MODE_WITHOUT,
    EXAMPLES_DIR,
    OUTPUT_UNITS,
    REQUIRED_INPUT_COLUMNS,
)
from gaia.inference import preload_models, run_inference
from gaia.validation import ValidationError

st.set_page_config(page_title="GAIA v2 - Geothermobarometry", page_icon="🌋", layout="wide")


@st.cache_resource(show_spinner="Caricamento dei modelli PyTorch...")
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
        "Formato file non supportato. Caricare un file .csv, .xls o .xlsx."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("GAIA v2")
st.subheader("Geo Artificial Intelligence thermobArometry")
st.write(
    "Rete neurale (PyTorch) per la stima di Pressione (kbar) e Temperatura (°C) "
    "dei sistemi di alimentazione vulcanica a partire dalla composizione chimica "
    "del clinopirosseno."
)
st.caption(
    "Versione standalone (GAIA_v2), basata sui modelli PyTorch definitivi. "
    "Vedi README.md per dettagli scientifici, limiti e attribuzioni."
)

with st.expander("Istruzioni e formato del file di input", expanded=False):
    st.markdown(
        "Il file caricato (csv, xls o xlsx) deve contenere le seguenti colonne, "
        "nell'ordine indicato:\n\n"
        f"`{', '.join(REQUIRED_INPUT_COLUMNS)}`\n\n"
        "- ***Index***, ***sample***, ***notes***, ***notes.1***: identificano il campione "
        "(qualsiasi testo).\n"
        "- Ossidi maggiori (wt%): ***SiO2, TiO2, Al2O3, Cr2O3, FeO tot, MnO, NiO, MgO, CaO, "
        "Na2O, K2O***. Se un ossido non è stato analizzato o è sotto il limite di rilevabilità, "
        "inserire 0 o lasciare la cella vuota.\n"
        "- ***tot***: totale analitico (informativo, non usato nel calcolo)."
    )
    example_path = EXAMPLES_DIR / "input_example.xlsx"
    if example_path.exists():
        with open(example_path, "rb") as f:
            st.download_button(
                "Scarica un file di esempio",
                data=f.read(),
                file_name="input_example.xlsx",
            )

# ---------------------------------------------------------------------------
# Chromium mode (explicit user choice - never inferred)
# ---------------------------------------------------------------------------
st.header("1. Modalità Cromo")
chromium_choice = st.radio(
    "Il Cromo (Cr2O3) è stato misurato nelle analisi caricate?",
    options=["Sì, il Cromo è stato misurato", "No, il Cromo non è stato misurato"],
    index=0,
)
chromium_mode = CHROMIUM_MODE_WITH if chromium_choice.startswith("Sì") else CHROMIUM_MODE_WITHOUT

if chromium_mode == CHROMIUM_MODE_WITHOUT:
    st.info(
        "Verrà utilizzata la famiglia di modelli addestrata azzerando le componenti "
        "cromo-dipendenti (CaCrTs, NaCrSi2O6), esattamente come in fase di training."
    )

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.header("2. Caricamento dati")
uploaded_file = st.file_uploader("Carica un file (.csv, .xls, .xlsx)", type=["csv", "xls", "xlsx"])

raw_df: pd.DataFrame | None = None
if uploaded_file is not None:
    try:
        raw_df = _read_uploaded_file(uploaded_file)
        st.write("Anteprima dei dati caricati:")
        st.dataframe(raw_df.head(20))
    except Exception:
        st.error(
            "Impossibile leggere il file caricato. Verificare che sia un csv/xls/xlsx valido "
            "e non vuoto."
        )
        raw_df = None

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
st.header("3. Predizione")

if raw_df is not None and st.button("Esegui pre-processing e predizione", type="primary"):
    try:
        _load_models_once()
        results = run_inference(raw_df, chromium_mode=chromium_mode)
    except ValidationError as e:
        st.error(f"Errore nei dati di input: {e}")
    except Exception:
        st.error(
            "Si è verificato un errore imprevisto durante l'elaborazione. "
            "Verificare il formato del file e riprovare."
        )
        with st.expander("Dettagli tecnici (per sviluppatori)"):
            st.code(traceback.format_exc())
    else:
        n_invalid = (~results["cpx_selection"]).sum()
        if n_invalid:
            st.warning(
                f"{n_invalid} campione/i su {len(results)} non superano i controlli di qualità "
                "chimica (cpx_selection) e sono marcati come non calcolabili (NaN)."
            )

        st.write("Predizioni (una riga per campione, stesso ordine dell'input):")
        display_cols = [
            "Index", "sample", "notes", "notes.1",
            "predicted_pressure", "predicted_pressure_std",
            "predicted_temperature", "predicted_temperature_std",
            "cpx_selection",
        ]
        st.dataframe(results[display_cols])
        st.caption(
            f"Unità: Pressione in {OUTPUT_UNITS['pressure']}, "
            f"Temperatura in {OUTPUT_UNITS['temperature']}. "
            "L'incertezza riportata è la deviazione standard tra i modelli dell'ensemble."
        )

        excel_bytes = _to_excel_bytes(results)
        st.download_button(
            "Scarica i risultati (xlsx)",
            data=excel_bytes,
            file_name="GAIA_v2_predictions.xlsx",
        )
elif raw_df is None:
    st.info("Carica un file per procedere con la predizione.")

