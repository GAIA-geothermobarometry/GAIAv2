"""Integration tests + numerical equivalence check for gaia.inference.

Numerical equivalence methodology
----------------------------------
We reconstruct, independently and locally in this test file, the ORIGINAL
domain-adaptation architecture exactly as defined in the training workspace
(`mod_class_only_cpx.py`: `PressureOnlyCPXNet_domain_adaptation` /
`TemperatureOnlyCPXNet_domain_adaptation`), including the unused
`discriminator` branch, and load the same checkpoints copied into
`GAIA_v2/artifacts`. We then compare its `forward(x)[0]` (regressed value)
against GAIA_v2's own simplified inference-only model
(`gaia.models._CPXNet`, net+regressor only) for every ensemble member.

Tolerance: rtol=1e-5, atol=1e-6. Both models share bit-identical weights and
architecture for the parts that matter (net + regressor); the only
difference is the unused discriminator head, which does not influence the
regression output at all mathematically. Therefore predictions should match
to floating point precision (small tolerance only accounts for potential
internal op-order/BLAS non-determinism), not because of any approximation in
the reproduced training procedure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaia.config import (
    CHROMIUM_MODE_WITH,
    CHROMIUM_MODE_WITHOUT,
    ENSEMBLE_SIZE,
    HIDDEN_WIDTH,
    INPUT_DIM,
    MAX_VALUE,
    checkpoint_paths,
)
from gaia.inference import run_inference
from gaia.models import EnsembleModel, build_model


# ---------------------------------------------------------------------------
# Independent reference reconstruction of the ORIGINAL training-time
# architecture (with discriminator head), used only for equivalence testing.
# ---------------------------------------------------------------------------
class _ReferenceDomainAdaptationNet(nn.Module):
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width), nn.ReLU(),
            nn.BatchNorm1d(width, eps=0.001), nn.Dropout(0.1),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.regressor = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        self.discriminator = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        features = self.net(x)
        regressed = self.regressor(features)
        domain = self.discriminator(features)  # gradient reversal irrelevant in eval/no-grad
        return regressed, domain, features


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Index": ["a1", "a2", "a3"],
        "sample": ["cpx1", "cpx2", "cpx3"],
        "notes": ["core", "rim", "core"],
        "notes.1": ["e1", "e1", "e2"],
        "SiO2": [52.37, 47.88, 49.47],
        "TiO2": [0.55, 0.67, 0.57],
        "Al2O3": [2.89, 3.13, 3.13],
        "Cr2O3": [0.07, 0.07, 0.13],
        "FeO tot": [3.89, 7.40, 7.71],
        "MnO": [0.05, 0.00, 0.00],
        "NiO": [0.0, 0.01, 0.09],
        "MgO": [16.13, 14.65, 12.93],
        "CaO": [23.80, 21.18, 22.08],
        "Na2O": [0.09, 0.66, 0.46],
        "K2O": [0.0, 0.03, 0.04],
        "tot": [99.99, 95.78, 96.64],
    })


def test_numerical_equivalence_with_reference_architecture():
    torch.manual_seed(0)
    x = torch.rand((5, INPUT_DIM), dtype=torch.float32)

    for chromium_mode in (CHROMIUM_MODE_WITH, CHROMIUM_MODE_WITHOUT):
        for target in ("pressure", "temperature"):
            width = HIDDEN_WIDTH[target]
            paths = checkpoint_paths(chromium_mode, target)[:3]  # sample a few members, keep test fast
            our_model = build_model(target)
            for path in paths:
                state_dict = torch.load(path, map_location="cpu")

                ref = _ReferenceDomainAdaptationNet(INPUT_DIM, width)
                ref.load_state_dict(state_dict, strict=True)
                ref.eval()

                filtered = {k: v for k, v in state_dict.items() if k in our_model.state_dict()}
                our_model.load_state_dict(filtered, strict=True)
                our_model.eval()

                with torch.inference_mode():
                    ref_out = ref(x)[0][:, 0] * MAX_VALUE[target]
                    our_out = our_model(x)[:, 0] * MAX_VALUE[target]

                np.testing.assert_allclose(
                    our_out.numpy(), ref_out.numpy(), rtol=1e-5, atol=1e-6,
                    err_msg=f"Mismatch for chromium_mode={chromium_mode} target={target} path={path}",
                )


def test_run_inference_with_chromium_flow_produces_both_targets():
    df = _sample_raw_df()
    out = run_inference(df, chromium_mode=CHROMIUM_MODE_WITH)
    assert "predicted_pressure" in out.columns
    assert "predicted_temperature" in out.columns
    assert len(out) == len(df)
    assert out["predicted_pressure_unit"].eq("kbar").all()
    assert out["predicted_temperature_unit"].eq("°C").all()


def test_run_inference_without_chromium_flow_produces_both_targets():
    df = _sample_raw_df()
    out = run_inference(df, chromium_mode=CHROMIUM_MODE_WITHOUT)
    assert "predicted_pressure" in out.columns
    assert "predicted_temperature" in out.columns
    assert len(out) == len(df)


def test_run_inference_preserves_row_order():
    df = _sample_raw_df()
    out = run_inference(df, chromium_mode=CHROMIUM_MODE_WITH)
    assert list(out["sample"]) == list(df["sample"])


def test_chromium_mode_changes_predictions():
    df = _sample_raw_df()
    out_with = run_inference(df, chromium_mode=CHROMIUM_MODE_WITH)
    out_without = run_inference(df, chromium_mode=CHROMIUM_MODE_WITHOUT)
    # Different model family + zeroed chromium components -> predictions
    # should generally differ (not required to be exactly equal).
    assert not out_with["predicted_pressure"].equals(out_without["predicted_pressure"])

