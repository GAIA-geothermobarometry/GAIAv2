# GAIA v2

Standalone application (PyTorch + Streamlit) for estimating **Pressure (kbar)**
and **Temperature (°C)** of volcano plumbing systems from clinopyroxene
chemical composition (geothermobarometry).

This is a self-contained rewrite of the previous GAIA app (see `GAIA_legacy/`
in the original development workspace), with a new inference layer based on
the final PyTorch models (July 2026), replacing the old TensorFlow models.

## What it does

1. The user uploads a file (csv/xls/xlsx) with clinopyroxene chemical
   analyses (major oxides in wt%).
2. The app computes the 11 structural components of the clinopyroxene (same
   chemical procedure as the legacy app).
3. An ensemble of PyTorch neural networks (100 models for pressure, 20 for
   temperature) predicts P and T for each sample, also reporting the
   uncertainty (standard deviation across the ensemble models).
4. Results can be downloaded as an Excel file, and prediction histograms can
   be plotted directly in the app.

## Measured vs. non-measured Chromium

There are **two model families**, trained separately:

- **`with_chromium`**: used when Cr2O3 was analyzed. The chromium-dependent
  components (`CaCrTs`, `NaCrSi2O6`) are computed normally.
- **`without_chromium`**: used when Chromium was NOT measured. **The models
  of this family were trained by zeroing the `CaCrTs` and `NaCrSi2O6`
  components** (after chemical preprocessing, before training/inference). To
  stay consistent with training, the app applies exactly the same zeroing
  procedure before invoking this model family.

Both flows share the **exact same chemical preprocessing pipeline**
(`gaia/preprocessing.py`); the only difference is the zeroing of the two
chromium-dependent components and the choice of checkpoint family.

The mode must **always be explicitly selected** by the user in the interface
(it is never automatically inferred from the data).

## Required input format

Required columns, in this order:

```
Index, sample, notes, notes.1, SiO2, TiO2, Al2O3, Cr2O3, FeO tot, MnO, NiO, MgO, CaO, Na2O, K2O, tot
```

- `Index`, `sample`, `notes`, `notes.1`: identify the sample (free text).
- Major oxides in wt%: if an oxide was not analyzed or is below the detection
  limit, enter `0` or leave the cell blank.
- `tot`: analytical total, informational column (not used in the computation).

An example file is available at `examples/input_example.xlsx` (derived from
the legacy app template) and an empty template at
`examples/input_template_empty.xlsx`.

## Installation

```powershell
conda create -n GAIAv2_env python=3.9
conda activate GAIAv2_env
pip install -r requirements.txt
```

## Local launch

```powershell
cd GAIA_v2
streamlit run app.py
```

The app is a Streamlit multipage app: the "Info" page (project background,
citation, people) is automatically available in the sidebar
(`pages/1_Info.py`).

## Tests

```powershell
cd GAIA_v2
python -m pytest tests -v
```

All tests (23) can be run from inside `GAIA_v2` without any dependency on
files or imports outside this folder. They include:

- shared, deterministic chemical preprocessing;
- correct zeroing of only the chromium-dependent components;
- input validation;
- checkpoint loading and `eval()` mode;
- end-to-end integration for both flows;
- **numerical equivalence**: the output of the simplified inference model
  (`gaia/models.py`, `net` + `regressor` only) is compared against an
  independent reconstruction of the full original architecture (including
  the `discriminator` branch, never used at inference time) loaded with the
  same weights. Tolerance used: `rtol=1e-5, atol=1e-6` (the weights and the
  relevant network part are bit-for-bit identical; the tolerance only covers
  possible internal BLAS/PyTorch non-determinism).

## Model organization (artifacts/)

```
artifacts/
├── with_chromium/
│   ├── pressure/       # 100 checkpoints (mod_0_.pth ... mod_99_.pth)
│   └── temperature/    # 20 checkpoints (mod_0_.pth ... mod_19_.pth)
└── without_chromium/
    ├── pressure/       # 100 checkpoints
    └── temperature/    # 20 checkpoints
```

Each `.pth` file is a plain PyTorch `state_dict` (no fully serialized model,
no embedded scaler). Total copied size: **~486.7 MB (240 files)**. Since this
exceeds the recommended threshold for a "normal" Git repository (~100 MB),
**Git LFS is recommended** for tracking the `artifacts/` folder if this
project is versioned in a Git repository. Git LFS has not been configured
automatically.

Not copied: `global_history.pkl` (training history) and the
`seq_sorted_bootstrap*.pickle` files (bootstrap indices), not needed for
inference.

## Model architecture

MLP with domain adaptation (only the `net` + `regressor` branch is used at
inference time; the `discriminator` branch, used during training for
adversarial domain adaptation, is ignored):

- Pressure: `Linear(11→100) → ReLU → BatchNorm1d → Dropout(0.1) → Linear(100→100) → ReLU → Linear(100→100) → ReLU → Linear(100→1)`
- Temperature: same structure with width 1000 instead of 100.

## Output and units

- **Pressure**: kbar (ensemble mean and standard deviation).
- **Temperature**: °C (ensemble mean and standard deviation).
- **`cpx_selection`**: boolean chemical quality flag (10 checks: Wo/En/Fs
  balance, sum of components, T site, M site, charge balance, etc. - same
  logic as the legacy app). Samples that fail the check have predictions set
  to `NaN` instead of a fabricated value.

## Important scientific assumptions

- No input scaler (StandardScaler/MinMax) is applied: the 11 components come
  from a chemical balance that already constrains them to a comparable range.
  This exactly reflects the original training pipeline.
- Output de-normalization is a simple fixed linear multiplication (`×10` for
  pressure, `×1400` for temperature), not a scaler fitted on data.
- The zeroed chromium-dependent components are exactly `CaCrTs` and
  `NaCrSi2O6` (and no other), applied after chemical preprocessing and before
  inference.

## Current limitations

- The app does not automatically check/validate whether the uploaded file is
  consistent with the selected Chromium mode (e.g. all Cr2O3 columns at zero
  while "Chromium measured" was selected): this choice remains the user's
  responsibility, by design (see requirements).
- No authentication, database, or multi-tenant deployment management: the
  app is designed for single-user/local use or deployment via Streamlit
  Community Cloud / an internal server.
- The models were validated on the datasets described in the original
  development workspace (GAIA1/GAIA2/Wieser); accuracy on chemical
  compositions very different from the training data is not guaranteed.

## Deployment on Streamlit Community Cloud

1. Version this folder (`GAIA_v2/`) in a dedicated Git repository
   (optionally with Git LFS for `artifacts/`, see above).
2. On [share.streamlit.io](https://share.streamlit.io), connect the
   repository and set `app.py` as the entry point.
3. Make sure `requirements.txt` is at the root of the repository (already
   present here).
4. If the deployment fails on dependency resolution (e.g. a `torch` build
   incompatible with the platform's default Python version), pin the Python
   version explicitly with a `.python-version` file at the repository root
   (this project uses Python 3.11).

## License and attribution

Based on the original GAIA project (Department of Physics and Astronomy and
Department of Earth Sciences, University of Florence). No explicit license
was indicated in the legacy app (`GAIA_legacy/GAIA-main/README.md` only
reports "[In publ.]" as a reference to the scientific publication). It is
recommended to verify/add an explicit license before any public
distribution.

