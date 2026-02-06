"""Unit tests for local few-shot models."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure `few_shot_learning` package is importable from repo root test runs.
PLUGIN_PARENT = Path(__file__).resolve().parents[2]
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

from few_shot_learning.models import (
    SUPPORTED_MODEL_NAMES,
    get_model,
)


def _make_binary_data(n_pos: int = 20, n_neg: int = 20, dim: int = 16, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos = rng.normal(loc=1.0, scale=0.4, size=(n_pos, dim)).astype(np.float32)
    neg = rng.normal(loc=-1.0, scale=0.4, size=(n_neg, dim)).astype(np.float32)
    embeddings = np.vstack([pos, neg])
    labels = np.array([1] * n_pos + [0] * n_neg, dtype=int)
    return embeddings, labels


def test_get_model_supported_names():
    for name in SUPPORTED_MODEL_NAMES:
        model = get_model(name)
        assert model is not None


def test_get_model_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("NotAModel")


def test_rocchio_fit_predict():
    embeddings, labels = _make_binary_data()
    model = get_model(
        "RocchioPrototypeModel",
        {"mode": "proto_softmax", "beta": 1.0, "gamma": 1.0, "temperature": 1.0},
    )
    model.fit_step([{"embeddings": embeddings, "labels": labels}])

    out = model.predict({"embeddings": embeddings[:8]})
    assert "scores" in out
    assert out["scores"].shape == (8,)
    assert np.all((out["scores"] >= 0.0) & (out["scores"] <= 1.0))


def test_rocchio_rejects_alpha_hyperparam():
    with pytest.raises(ValueError, match="does not support 'alpha'"):
        get_model("RocchioPrototypeModel", {"alpha": 0.5})


def test_stack_embeddings_labels_rejects_mixed_batches():
    from few_shot_learning.models import _stack_embeddings_and_labels

    emb = np.random.randn(4, 8).astype(np.float32)
    with pytest.raises(ValueError, match="Mixed batches"):
        _stack_embeddings_and_labels(
            [{"embeddings": emb, "labels": np.array([0, 1, 0, 1])}, {"embeddings": emb}],
            require_labels=False,
        )
