"""GAIA_v2 - Info page.

Adapted from the legacy GAIA app (GAIA_legacy/GAIA-main/pages/Info.py),
translated to English and updated for the GAIA_v2 standalone project.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGO_PATH = PROJECT_ROOT / "logo_gaiav2.png"
_logo_image = Image.open(LOGO_PATH) if LOGO_PATH.exists() else None

st.set_page_config(
    page_title="GAIA 2 - Info",
    page_icon=_logo_image if _logo_image is not None else "🌋",
    layout="wide",
)

st.title("GAIA  - About the project")

col_title, col_logo = st.columns([3, 1])
with col_title:
    st.write(
        "The anatomy of the plumbing system of active volcanoes is fundamental to "
        "understanding how magma is stored and channeled to the surface. Reliable "
        "geothermobarometric estimates are, therefore, critical to assess the depths "
        "and temperatures of the complex system of magmatic reservoirs that form a "
        "volcano apparatus. Here, we developed a novel Machine Learning approach based "
        "upon Feedforward Neural Networks (GAIA) to estimate P-T conditions of magma "
        "storage and migration within the crust. Our Feedforward Neural Network method "
        "applied to clinopyroxene compositions yields better uncertainties "
        "(Root-Mean-Square Error and R2 score) than previous Machine Learning methods "
        "and sets the basis for a novel generation of reliable geothermobarometers, "
        "which extends beyond the paradigm associated with crystal-liquid equilibrium. "
        "Also, the bootstrap/ensemble procedure, inherent to the Feedforward Neural "
        "Network architecture, permits a rigorous assessment of the P-T uncertainty "
        "associated with each clinopyroxene composition, as opposed to the "
        "Root-Mean-Square Error representing the P-T uncertainty of the whole set of "
        "clinopyroxene compositions."
    )

with col_logo:
    if _logo_image is not None:
        st.image(_logo_image, width=250)

st.header("About GAIA 2")
st.write(
    "GAIA 2 is a fine-tuned evolution of the original GAIA model, trained on a substantially broader range of experimental laboratory data. It also incorporates natural samples during calibration through a domain-adaptation procedure, making it, to our knowledge, the first machine-learning model for geothermobarometry to use natural data as part of the training process. The standalone web application uses the final PyTorch ensemble models for pressure and temperature prediction while retaining the established chemical preprocessing and scientific validation framework. See the repository’s README.md for full technical details, methodological assumptions, and limitations."
)

st.header("Citing")
st.write("To cite GAIA please use the following publication:")
st.write(
    "Chicchi, L., Bindi, L., Fanelli, D., & Tommasini, S. (2023). Frontiers of "
    "thermobarometry: GAIA, a novel Deep Learning-based tool for volcano plumbing "
    "systems. Earth and Planetary Science Letters, 620, 118352."
)
url = (
    "https://www.sciencedirect.com/science/article/pii/S0012821X23003655"
)
st.write("Paper [here](%s)" % url)

st.header("People behind GAIA")
st.markdown(
    "- **Dr. Lorenzo Chicchi**, Università degli Studi di Firenze, Dipartimento di "
    "Fisica e Astronomia, INFN\n"
    "- **Prof. Luca Bindi**, Università degli Studi di Firenze, Dipartimento di "
    "Scienze della Terra\n"
    "- **Prof. Duccio Fanelli**, Università degli Studi di Firenze, Dipartimento "
    "di Fisica e Astronomia, INFN\n"
    "- **Dr. Diego Febbe**, Università degli Studi di Firenze, Dipartimento "
    "di Fisica e Astronomia, INFN\n"
     "- **Dr. Gianluca Peri**, Università degli Studi di Firenze, Dipartimento "
    "di Fisica e Astronomia, INFN\n"
    "- **Prof. Simone Tommasini**, Università degli Studi di Firenze, Dipartimento "
    "di Scienze della Terra"
)

