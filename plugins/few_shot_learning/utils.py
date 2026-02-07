"""Utility classes and functions for few-shot learning."""

import numpy as np
import torch
from fiftyone.utils.torch import GetItem


class EmbeddingsGetItem(GetItem):
    """GetItem that extracts embeddings and sample IDs for DataLoader inference."""

    @property
    def required_keys(self) -> list[str]:
        """Return fields required from each sample."""
        return ["embeddings", "id"]

    def __call__(self, d: dict) -> dict:
        """Transform sample dict to model input format."""
        embeddings = d["embeddings"]
        sample_id = str(d["id"])

        # Convert embeddings to float32 tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        elif isinstance(embeddings, (list, tuple)):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.float()

        return {"embeddings": embeddings, "id": sample_id}


def collate_fn(batch: list) -> dict:
    """Custom collate function that stacks embeddings and preserves IDs.

    Filters out any Exception objects that may be present when skip_failures=True.
    """
    # Filter out exceptions (skip_failures=True can yield Exception objects)
    valid_items = [item for item in batch if isinstance(item, dict)]

    if not valid_items:
        # Return empty batch if all items failed
        return {"embeddings": torch.empty(0), "ids": []}

    embeddings = torch.stack([item["embeddings"] for item in valid_items])
    ids = [item["id"] for item in valid_items]
    return {"embeddings": embeddings, "ids": ids}


def extract_probability(batch: dict, model_name: str) -> np.ndarray:
    """
    Extract P(positive) from model output batch using priority order:
    1. predictions with Classification type or dict -> use confidence for "positive" label
    2. p_pos -> use directly
    3. probs -> use scalar or column 1 as P(positive)
    4. scores -> use directly for Rocchio (already [0,1]), apply sigmoid for NCA/LMNN

    Args:
        batch: Model output dictionary
        model_name: Name of the model (used to determine score handling)

    Returns:
        np.ndarray of probabilities with shape (batch_size,)
    """
    # Priority 1: Check for predictions (OutputProcessor results)
    if "predictions" in batch:
        preds = batch["predictions"]
        if isinstance(preds, (list, tuple)) and len(preds) > 0:
            first = preds[0]
            # Handle FiftyOne Classification objects
            if hasattr(first, "label") and hasattr(first, "confidence"):
                probs = []
                for p in preds:
                    if p.label == "positive":
                        probs.append(p.confidence)
                    else:
                        probs.append(1.0 - p.confidence)
                return np.array(probs)
            # Handle dict predictions with _type: "classification"
            if isinstance(first, dict):
                probs = []
                for p in preds:
                    label = p.get("label", "negative")
                    confidence = p.get("confidence", 0.5)
                    if label == "positive":
                        probs.append(confidence)
                    else:
                        probs.append(1.0 - confidence)
                return np.array(probs)

    # Priority 2: p_pos (GraphLabelPropagationModel)
    if "p_pos" in batch:
        p_pos = batch["p_pos"]
        if isinstance(p_pos, torch.Tensor):
            return p_pos.cpu().numpy()
        return np.asarray(p_pos)

    # Priority 3: probs (LinearSVMModel)
    if "probs" in batch:
        probs = batch["probs"]
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        else:
            probs = np.asarray(probs)
        # Handle 2D probs (column 1 is positive class)
        if probs.ndim == 2:
            return probs[:, 1]
        return probs

    # Priority 4: scores
    if "scores" in batch:
        scores = batch["scores"]
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        else:
            scores = np.asarray(scores)

        # Rocchio scores are already in [0, 1] - use directly
        if model_name == "RocchioPrototypeModel":
            return np.clip(scores, 0.0, 1.0)

        # NCA/LMNN scores are unbounded (dist_neg - dist_pos) - apply sigmoid
        return 1.0 / (1.0 + np.exp(-scores))

    raise ValueError(
        "Model output missing expected keys. "
        f"Got: {list(batch.keys())}. "
        "Expected one of: predictions, p_pos, probs, scores"
    )
