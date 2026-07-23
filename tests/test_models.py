"""Tests for gaia.models: checkpoint loading, eval mode, ensemble sizes."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaia.config import CHROMIUM_MODES, ENSEMBLE_SIZE, HIDDEN_WIDTH, INPUT_DIM, TARGETS
from gaia.models import EnsembleModel, build_model


def test_build_model_shapes():
    for target in TARGETS:
        m = build_model(target)
        width = HIDDEN_WIDTH[target]
        assert m.net[0].in_features == INPUT_DIM
        assert m.net[0].out_features == width


def test_ensemble_loads_all_checkpoints_and_eval_mode():
    for chromium_mode in CHROMIUM_MODES:
        for target in TARGETS:
            ens = EnsembleModel(target=target, chromium_mode=chromium_mode)
            assert len(ens) == ENSEMBLE_SIZE[target]
            for m in ens.models:
                assert m.training is False


def test_ensemble_predict_raw_shape():
    ens = EnsembleModel(target="pressure", chromium_mode="with_chromium")
    x = torch.zeros((3, INPUT_DIM), dtype=torch.float32)
    out = ens.predict_raw(x)
    assert out.shape == (ENSEMBLE_SIZE["pressure"], 3)

