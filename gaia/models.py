"""PyTorch model definitions and ensemble checkpoint loading for GAIA_v2.

Architectures are a direct, unmodified port of the domain-adaptation MLPs
defined in ``mod_class_only_cpx.py`` (original training workspace):
``PressureOnlyCPXNet_domain_adaptation`` / ``TemperatureOnlyCPXNet_domain_adaptation``.

For inference we only need the ``net`` (shared backbone) and ``regressor``
head; the ``discriminator`` head (used only for adversarial domain
adaptation during training) is intentionally NOT instantiated here. Because
of that, checkpoints (which do contain ``discriminator.*`` keys) are loaded
with ``strict=False`` and the unexpected discriminator keys are verified and
discarded explicitly (see :func:`_load_state_dict_backbone_only`).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import HIDDEN_WIDTH, INPUT_DIM, TARGETS, checkpoint_paths


class _CPXNet(nn.Module):
    """Shared backbone + regressor head (inference-only subset of the
    original domain-adaptation network)."""

    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.BatchNorm1d(width, eps=0.001),
            nn.Dropout(0.1),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return self.regressor(features)


def build_model(target: str) -> _CPXNet:
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}', expected one of {TARGETS}")
    return _CPXNet(INPUT_DIM, HIDDEN_WIDTH[target])


def _load_state_dict_backbone_only(model: _CPXNet, checkpoint_path: Path) -> None:
    """Load a checkpoint state_dict onto ``model``, keeping only the keys
    the inference model actually has (``net.*`` and ``regressor.*``), and
    verifying that no expected key is missing.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint '{checkpoint_path}' is not a state_dict (found {type(state_dict)!r}); "
            "unsupported checkpoint format."
        )
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    if missing:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' is missing expected keys: {sorted(missing)}"
        )
    model.load_state_dict(filtered, strict=True)


class EnsembleModel:
    """A bootstrap ensemble of identical architectures for one (chromium_mode,
    target) pair. Prediction = mean over ensemble members, uncertainty =
    standard deviation over ensemble members (both in physical output
    units, computed by :mod:`gaia.inference`)."""

    def __init__(self, target: str, chromium_mode: str):
        self.target = target
        self.chromium_mode = chromium_mode
        paths = checkpoint_paths(chromium_mode, target)
        missing_files = [p for p in paths if not p.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing {len(missing_files)} checkpoint(s) for target='{target}' "
                f"chromium_mode='{chromium_mode}', e.g. {missing_files[0]}"
            )
        self.models: list[_CPXNet] = []
        for p in paths:
            m = build_model(target)
            _load_state_dict_backbone_only(m, p)
            m.eval()
            self.models.append(m)

    def __len__(self) -> int:
        return len(self.models)

    @torch.inference_mode()
    def predict_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw (non-denormalized) predictions, shape
        (n_ensemble_members, n_samples)."""
        outputs = [m(x)[:, 0] for m in self.models]
        return torch.stack(outputs, dim=0)

