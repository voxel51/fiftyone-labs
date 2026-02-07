"""Self-contained few-shot models used by the panel."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


SUPPORTED_MODEL_NAMES = [
    "RocchioPrototypeModel",
]

BatchDict = dict[str, Any]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _as_float32_2d_embeddings(embeddings: Any) -> np.ndarray:
    """Convert input embeddings to a `(N, D)` float32 NumPy array."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (N, D), got {arr.shape}")
    return arr


def _normalize_binary_labels(labels: Any, model_name: str) -> np.ndarray:
    """Normalize binary labels to `{0, 1}` where `1` denotes positives."""
    y = np.asarray(labels).astype(int).reshape(-1)
    uniq = set(np.unique(y).tolist())

    if uniq.issubset({0, 1}):
        return y
    if uniq.issubset({-1, 1}):
        return (y == 1).astype(int)

    raise ValueError(
        f"{model_name} expects binary labels in {{0,1}} or {{-1,+1}}, got {sorted(uniq)}"
    )


def _stack_embeddings_and_labels(
    batches: Sequence[BatchDict], *, require_labels: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Concatenate embeddings and labels across a sequence of batches."""
    if not batches:
        raise ValueError("Expected at least one batch")

    embeddings_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    labels_missing = False

    for batch in batches:
        if "embeddings" not in batch:
            raise KeyError("Batch is missing 'embeddings' key.")

        embeddings = _as_float32_2d_embeddings(batch["embeddings"])
        embeddings_list.append(embeddings)

        labels = batch.get("labels", None)
        if labels is None:
            labels_missing = True
            continue

        labels_arr = np.asarray(labels).reshape(-1)
        if len(labels_arr) != len(embeddings):
            raise ValueError(
                "Batch labels length must match embeddings rows: "
                f"{len(labels_arr)} != {len(embeddings)}"
            )
        labels_list.append(labels_arr)

    stacked_embeddings = np.vstack(embeddings_list)

    if labels_missing:
        if require_labels:
            raise ValueError("Labels are required for this model")
        if labels_list:
            raise ValueError(
                "Mixed batches with and without labels are not supported. "
                "Provide labels for all batches or none."
            )
        return stacked_embeddings, None

    if not labels_list:
        if require_labels:
            raise ValueError("Labels are required for this model")
        return stacked_embeddings, None

    stacked_labels = np.concatenate(labels_list)
    if len(stacked_labels) != len(stacked_embeddings):
        raise ValueError(
            "Concatenated labels length must match concatenated embeddings rows: "
            f"{len(stacked_labels)} != {len(stacked_embeddings)}"
        )
    return stacked_embeddings, stacked_labels


class _BaseModel:
    """Minimal model interface used by the panel training loop."""

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """Store hyperparameters and initialize compatibility flags."""
        self.hyperparams = dict(hyperparams or {})
        self.enable_postprocess = False

    def build_output_processor(self, classes: list[str] | None = None) -> None:
        """Compatibility no-op for callers expecting this method."""
        # Panel calls this opportunistically; keep it as a compatibility no-op.
        self._output_classes = list(classes or [])

    def predict(self, batch: BatchDict) -> BatchDict:
        """Run model inference on a single batch dictionary."""
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got {type(batch)!r}")
        return self._predict_dict(dict(batch))

    def _predict_dict(self, batch: BatchDict) -> BatchDict:
        """Internal predict implementation overridden by subclasses."""
        raise NotImplementedError


class RocchioPrototypeModel(_BaseModel):
    """Centroid-based binary scorer with Rocchio/prototype variants."""

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """Initialize Rocchio scoring configuration and state."""
        super().__init__(hyperparams)
        self._fitted = False

        if "alpha" in self.hyperparams:
            raise ValueError(
                "RocchioPrototypeModel does not support 'alpha' in this plugin port."
            )

        self._mode = str(self.hyperparams.get("mode", "proto_softmax"))
        self._beta = float(self.hyperparams.get("beta", 1.0))
        self._gamma = float(self.hyperparams.get("gamma", 1.0))
        self._temperature = float(self.hyperparams.get("temperature", 1.0))
        self._normalize_embeddings = bool(self.hyperparams.get("normalize_embeddings", True))

        self._pos_centroid: np.ndarray | None = None
        self._neg_centroid: np.ndarray | None = None
        self._query_vector: np.ndarray | None = None

    def _transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply optional L2 normalization to embeddings."""
        if not self._normalize_embeddings:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms

    def fit_step(self, batches: Sequence[BatchDict]) -> None:
        """Fit class centroids from labeled embedding batches."""
        embeddings, labels = _stack_embeddings_and_labels(batches, require_labels=True)
        labels = _normalize_binary_labels(labels, "RocchioPrototypeModel")

        if len(np.unique(labels)) < 2:
            raise ValueError(
                "RocchioPrototypeModel requires at least 2 classes. "
                "Ensure you have both positive and negative examples."
            )

        transformed = self._transform(embeddings)
        pos = transformed[labels == 1]
        neg = transformed[labels == 0]

        self._pos_centroid = pos.mean(axis=0)
        self._neg_centroid = neg.mean(axis=0)
        self._query_vector = self._beta * self._pos_centroid - self._gamma * self._neg_centroid
        self._fitted = True

    def _predict_dict(self, batch: BatchDict) -> BatchDict:
        """Predict Rocchio/prototype confidence scores in `[0, 1]`."""
        if "embeddings" not in batch:
            raise KeyError("Batch is missing 'embeddings' key.")

        embeddings = _as_float32_2d_embeddings(batch["embeddings"])
        transformed = self._transform(embeddings)
        batch["embeddings_transformed"] = transformed

        if not self._fitted:
            batch["scores"] = np.zeros(len(embeddings), dtype=np.float32)
            return batch

        temp = max(self._temperature, 1e-8)
        if self._mode == "proto_softmax":
            dist_to_pos = np.linalg.norm(transformed - self._pos_centroid, axis=1)
            dist_to_neg = np.linalg.norm(transformed - self._neg_centroid, axis=1)
            scores = _sigmoid((dist_to_neg - dist_to_pos) / temp)
        elif self._mode == "rocchio_sigmoid":
            raw = transformed @ self._query_vector
            scores = _sigmoid(raw / temp)
        else:
            raise ValueError(f"Unknown Rocchio mode: {self._mode}")

        batch["scores"] = np.asarray(scores, dtype=np.float32)
        return batch

    @property
    def is_fitted(self) -> bool:
        """Whether `fit_step` has completed successfully."""
        return self._fitted


def get_model(name: str, hyperparams: dict[str, Any] | None = None) -> _BaseModel:
    """Instantiate one of the plugin's built-in few-shot model classes."""
    hyperparams = dict(hyperparams or {})

    if name == "RocchioPrototypeModel":
        return RocchioPrototypeModel(hyperparams)

    raise ValueError(
        f"Unknown model: '{name}'. Available models: {SUPPORTED_MODEL_NAMES}"
    )


__all__ = [
    "SUPPORTED_MODEL_NAMES",
    "get_model",
    "RocchioPrototypeModel",
]
