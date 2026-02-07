"""Few-Shot Learning Panel for FiftyOne Labs."""

import contextlib
import hashlib
import io
import json
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .models import get_model

# Supported classification models for few-shot learning
SUPPORTED_MODELS = [
    "RocchioPrototypeModel",
]

# Embedding models available from the FiftyOne model zoo
EMBEDDING_ZOO_MODELS = {
    "ResNet18": "resnet18-imagenet-torch",
    "ResNet50": "resnet50-imagenet-torch",
    "CLIP (ViT-B/32)": "clip-vit-base32-torch",
    "DINOv2 (ViT-B/14)": "dinov2-vitb14-torch",
}

EMBEDDING_FIELD_DEFAULTS = {
    "ResNet18": "resnet18_embeddings",
    "ResNet50": "resnet50_embeddings",
    "CLIP (ViT-B/32)": "clip_vit_b32_embeddings",
    "DINOv2 (ViT-B/14)": "dinov2_vitb14_embeddings",
}

EMBEDDING_DIMENSIONS = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "CLIP (ViT-B/32)": 512,
    "DINOv2 (ViT-B/14)": 768,
}

# Model hyperparameters schema: name -> {param: (default, description, type, [choices])}
# type: "int", "float", "str", "choice"
MODEL_HYPERPARAMS = {
    "RocchioPrototypeModel": {
        "mode": (
            "proto_softmax",
            "Scoring mode: proto_softmax = nearest centroid; "
            "rocchio_sigmoid = query vector dot product",
            "choice",
            ["proto_softmax", "rocchio_sigmoid"],
        ),
        "beta": (1.0, "Weight for positive centroid", "float"),
        "gamma": (1.0, "Weight for negative centroid", "float"),
        "temperature": (1.0, "Temperature for softmax/sigmoid", "float"),
        "normalize_embeddings": (
            True,
            "L2-normalize embeddings before scoring",
            "bool",
        ),
    },
}

from .utils import EmbeddingsGetItem, collate_fn, extract_probability


def _view_fingerprint(ids: list[str]) -> str:
    """Deterministic fingerprint of a set of sample IDs."""
    data = "\n".join(sorted(ids)).encode()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


@dataclass
class FewShotSession:
    """Tracks few-shot learning session state."""

    embedding_model: str = "ResNet18"
    embedding_field: str = "resnet18_embeddings"
    label_field: str = "fewshot_prediction"

    # Model configuration
    model_name: str = "RocchioPrototypeModel"
    model_hyperparams: dict[str, Any] = field(default_factory=dict)

    # DataLoader settings
    batch_size: int = 16
    num_workers: int = 8
    skip_failures: bool = True

    # Session state
    positive_ids: list[str] = field(default_factory=list)
    negative_ids: list[str] = field(default_factory=list)
    iteration: int = 0
    train_positive_field: str = "train_positive"
    train_negative_field: str = "train_negative"

    # Subset sampling settings
    working_subset_size: int = 0  # 0 means no limit (use all samples)
    randomize_subset: bool = False  # Re-sample each iteration
    subset_ids: list[str] = field(
        default_factory=list
    )  # Cached subset IDs when not randomizing
    subset_source_fingerprint: str = ""  # Fingerprint of source view IDs

    def add_positives(self, ids: list[str]) -> None:
        """Add IDs to positive set (retained across iterations)."""
        for sample_id in ids:
            if sample_id not in self.positive_ids:
                self.positive_ids.append(sample_id)
            if sample_id in self.negative_ids:
                self.negative_ids.remove(sample_id)

    def add_negatives(self, ids: list[str]) -> None:
        """Add IDs to negative set (retained across iterations)."""
        for sample_id in ids:
            if sample_id not in self.negative_ids:
                self.negative_ids.append(sample_id)
            if sample_id in self.positive_ids:
                self.positive_ids.remove(sample_id)

    def can_train(self) -> bool:
        """Return whether minimum labels exist to run training."""
        return len(self.positive_ids) >= 1 and len(self.negative_ids) >= 1

    def get_stats(self) -> dict[str, int]:
        """Return simple session counters for the panel header."""
        return {
            "iteration": self.iteration,
            "positives": len(self.positive_ids),
            "negatives": len(self.negative_ids),
        }


class FewShotLearningPanel(foo.Panel):
    """Interactive panel for few-shot learning with multiple model types."""

    @property
    def config(self) -> foo.PanelConfig:
        """Panel registration metadata."""
        return foo.PanelConfig(
            name="few_shot_learning",
            label="Few-Shot Learning",
        )

    def _get_session(self, ctx: Any) -> Optional[FewShotSession]:
        """Get current session from panel state."""
        try:
            session_data = ctx.panel.state.session
            if session_data is None:
                return None

            def _get(key: str, default: Any) -> Any:
                """Read values from either dict-like or attribute-like state."""
                if isinstance(session_data, dict):
                    return session_data.get(key, default)
                return getattr(session_data, key, default)

            # Parse model_hyperparams from JSON string if needed
            hyperparams = _get("model_hyperparams", {})
            if isinstance(hyperparams, str):
                try:
                    hyperparams = (
                        json.loads(hyperparams) if hyperparams else {}
                    )
                except json.JSONDecodeError:
                    hyperparams = {}

            return FewShotSession(
                embedding_model=_get("embedding_model", "ResNet18"),
                embedding_field=_get("embedding_field", "resnet18_embeddings"),
                label_field=_get("label_field", "fewshot_prediction"),
                model_name=_get("model_name", "RocchioPrototypeModel"),
                model_hyperparams=hyperparams,
                batch_size=int(_get("batch_size", 16)),
                num_workers=int(_get("num_workers", 8)),
                skip_failures=bool(_get("skip_failures", True)),
                positive_ids=list(_get("positive_ids", [])),
                negative_ids=list(_get("negative_ids", [])),
                iteration=int(_get("iteration", 0)),
                train_positive_field=str(
                    _get("train_positive_field", "train_positive")
                ),
                train_negative_field=str(
                    _get("train_negative_field", "train_negative")
                ),
                working_subset_size=int(_get("working_subset_size", 0)),
                randomize_subset=bool(_get("randomize_subset", False)),
                subset_ids=list(_get("subset_ids", [])),
                subset_source_fingerprint=str(
                    _get("subset_source_fingerprint", "")
                ),
            )
        except (AttributeError, TypeError):
            return None

    def _save_session(self, ctx: Any, session: FewShotSession) -> None:
        """Save session to panel state."""
        ctx.panel.state.session = {
            "embedding_model": session.embedding_model,
            "embedding_field": session.embedding_field,
            "label_field": session.label_field,
            "model_name": session.model_name,
            "model_hyperparams": session.model_hyperparams,
            "batch_size": session.batch_size,
            "num_workers": session.num_workers,
            "skip_failures": session.skip_failures,
            "positive_ids": session.positive_ids,
            "negative_ids": session.negative_ids,
            "iteration": session.iteration,
            "train_positive_field": session.train_positive_field,
            "train_negative_field": session.train_negative_field,
            "working_subset_size": session.working_subset_size,
            "randomize_subset": session.randomize_subset,
            "subset_ids": session.subset_ids,
            "subset_source_fingerprint": session.subset_source_fingerprint,
        }

    def _clear_session(self, ctx: Any) -> None:
        """Clear session from panel state."""
        ctx.panel.state.session = None

    def on_change_embedding_model(self, ctx: Any) -> None:
        """Update embedding field name when model changes."""
        model = (
            ctx.params.get("value")
            or getattr(ctx.panel.state, "embedding_model", None)
            or "ResNet18"
        )
        current_field = getattr(ctx.panel.state, "embedding_field", None) or ""
        known_defaults = set(EMBEDDING_FIELD_DEFAULTS.values())
        if not current_field or current_field in known_defaults:
            ctx.panel.state.embedding_field = EMBEDDING_FIELD_DEFAULTS.get(
                model, "embeddings"
            )

    def _render_hyperparams(
        self, panel: types.Object, model_name: str
    ) -> None:
        """Render individual hyperparameter fields for the selected model."""
        schema = MODEL_HYPERPARAMS.get(model_name, {})

        if not schema:
            return

        panel.md("**Model Hyperparameters:**")

        for param_name, param_info in schema.items():
            default_val = param_info[0]
            description = param_info[1]
            field_type = param_info[2] if len(param_info) > 2 else "float"

            field_key = f"hyperparam_{param_name}"

            if field_type == "int":
                panel.int(
                    field_key,
                    label=param_name,
                    default=default_val,
                    description=description,
                )
            elif field_type == "float":
                panel.float(
                    field_key,
                    label=param_name,
                    default=default_val,
                    description=description,
                )
            elif field_type == "bool":
                panel.bool(
                    field_key,
                    label=param_name,
                    default=default_val,
                    description=description,
                )
            elif field_type == "choice":
                choices = param_info[3]
                dropdown = types.DropdownView()
                for choice in choices:
                    dropdown.add_choice(choice, label=choice)
                panel.str(
                    field_key,
                    label=param_name,
                    view=dropdown,
                    default=default_val,
                    description=description,
                )
            else:  # str
                panel.str(
                    field_key,
                    label=param_name,
                    default=str(default_val),
                    description=description,
                )

    def _collect_hyperparams(
        self, ctx: Any, model_name: str
    ) -> dict[str, Any]:
        """Collect hyperparameter values from panel state."""
        schema = MODEL_HYPERPARAMS.get(model_name, {})
        hyperparams: dict[str, Any] = {}

        for param_name, param_info in schema.items():
            default_val = param_info[0]
            field_key = f"hyperparam_{param_name}"
            value = getattr(ctx.panel.state, field_key, None)

            if value is not None:
                hyperparams[param_name] = value
            else:
                hyperparams[param_name] = default_val

        return hyperparams

    def _ensure_embeddings(
        self, ctx: Any, view: Any, field_name: str, zoo_model_name: str
    ) -> int:
        """Compute missing embeddings. Returns number of samples computed."""
        need_compute = view.exists(field_name, False)
        count = len(need_compute)
        if count == 0:
            return 0

        ctx.ops.notify(
            f"Computing {field_name} for {count} samples...",
            variant="info",
        )
        try:
            model = foz.load_zoo_model(zoo_model_name)
            # Redirect stdout: FiftyOne's compute_embeddings uses an ETA
            # progress bar that writes to sys.stdout, which may be closed
            # in the operator thread-pool context.
            with contextlib.redirect_stdout(io.StringIO()):
                need_compute.compute_embeddings(
                    model,
                    embeddings_field=field_name,
                    batch_size=32,
                    num_workers=4,
                    skip_failures=True,
                )
        except Exception as e:
            ctx.ops.notify(
                f"Failed to compute embeddings in '{field_name}': {e}",
                variant="error",
            )
            raise
        return count

    def _ensure_training_label_fields(
        self, ctx: Any, session: FewShotSession
    ) -> None:
        """Ensure session-scoped training label fields exist."""
        schema = ctx.dataset.get_field_schema()
        for field_name in (
            session.train_positive_field,
            session.train_negative_field,
        ):
            if field_name not in schema:
                ctx.dataset.add_sample_field(field_name, fo.BooleanField)

    def _check_embedding_dimension(
        self, ctx: Any, field_name: str, embedding_model: str
    ) -> bool:
        """Return False and notify if existing embeddings have wrong dim."""
        expected_dim = EMBEDDING_DIMENSIONS.get(embedding_model)
        if not expected_dim:
            return True

        schema = ctx.dataset.get_field_schema()
        if field_name not in schema:
            return True

        sample_view = ctx.dataset.exists(field_name).limit(20)
        if len(sample_view) == 0:
            return True

        for sample in sample_view:
            emb = sample[field_name]
            if emb is not None:
                actual_dim = np.asarray(emb).shape[-1]
                if actual_dim != expected_dim:
                    ctx.ops.notify(
                        f"Field '{field_name}' has dimension {actual_dim} "
                        f"but {embedding_model} expects {expected_dim}. "
                        f"Choose a different field name or matching model.",
                        variant="error",
                    )
                    return False
        return True

    def _get_inference_view(
        self, ctx: Any, session: FewShotSession
    ) -> tuple[Any, int]:
        """Get the view to use for inference, applying subset sampling.

        Samples from ctx.view (respects filters/slices). When subset
        sampling is active, labeled samples are always included even if
        outside the current view.

        Returns (view, count) tuple.
        """
        import random

        source = ctx.view

        # No subset limit
        if session.working_subset_size <= 0:
            return source, len(source)

        # Labeled IDs always included
        labeled_ids = set(session.positive_ids + session.negative_ids)

        # Compute view fingerprint to detect filter changes
        source_ids = source.values("id")
        fingerprint = _view_fingerprint(source_ids)
        source_id_set = set(source_ids)

        # Cached subset (not randomizing) — invalidate if view changed
        if not session.randomize_subset and session.subset_ids:
            if fingerprint == session.subset_source_fingerprint:
                subset_ids = set(session.subset_ids) | labeled_ids
                return ctx.dataset.select(list(subset_ids)), len(subset_ids)
            # View changed, invalidate cache
            session.subset_ids = []

        # Sample unlabeled from current view
        unlabeled_ids = list(source_id_set - labeled_ids)

        unlabeled_limit = max(
            0, session.working_subset_size - len(labeled_ids)
        )

        if len(unlabeled_ids) <= unlabeled_limit:
            sampled_unlabeled = unlabeled_ids
        else:
            sampled_unlabeled = random.sample(unlabeled_ids, unlabeled_limit)

        subset_ids = list(labeled_ids) + sampled_unlabeled

        if not session.randomize_subset:
            session.subset_ids = subset_ids
            session.subset_source_fingerprint = fingerprint

        return ctx.dataset.select(subset_ids), len(subset_ids)

    def on_load(self, ctx: Any) -> None:
        """Initialize panel state."""
        pass

    def on_change_selected(self, ctx: Any) -> None:
        """Capture sample selection changes.

        Note: We intentionally do NOT update panel state here to avoid
        triggering a re-render that resets the grid scroll position.
        Instead, we read ctx.selected directly when needed.
        """
        pass

    def start_session(self, ctx: Any) -> None:
        """Start a new few-shot learning session."""
        embedding_model = (
            getattr(ctx.panel.state, "embedding_model", None) or "ResNet18"
        )
        embedding_field = getattr(
            ctx.panel.state, "embedding_field", None
        ) or EMBEDDING_FIELD_DEFAULTS.get(embedding_model, "embeddings")
        batch_size = getattr(ctx.panel.state, "batch_size", None) or 16
        num_workers = getattr(ctx.panel.state, "num_workers", None)
        if num_workers is None:
            num_workers = 8
        skip_failures = getattr(ctx.panel.state, "skip_failures", True)

        # Subset sampling settings
        working_subset_size = (
            getattr(ctx.panel.state, "working_subset_size", None) or 0
        )
        randomize_subset = getattr(ctx.panel.state, "randomize_subset", False)

        # Check dimension compatibility before computing
        if not self._check_embedding_dimension(
            ctx, embedding_field, embedding_model
        ):
            return

        session = FewShotSession(
            embedding_model=embedding_model,
            embedding_field=embedding_field,
            label_field="fewshot_prediction",
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            skip_failures=bool(skip_failures),
            working_subset_size=int(working_subset_size),
            randomize_subset=bool(randomize_subset),
            train_positive_field=f"fewshot_train_positive_{uuid.uuid4().hex[:8]}",
            train_negative_field=f"fewshot_train_negative_{uuid.uuid4().hex[:8]}",
        )

        # Create session-scoped training fields once per session so
        # they are ephemeral and cleaned up by reset_session.
        self._ensure_training_label_fields(ctx, session)

        # Compute embeddings for inference view
        inference_view, _ = self._get_inference_view(ctx, session)
        zoo_model_name = EMBEDDING_ZOO_MODELS[embedding_model]
        computed = self._ensure_embeddings(
            ctx, inference_view, embedding_field, zoo_model_name
        )

        self._save_session(ctx, session)

        msg = "Session started! Select samples and label them."
        if computed > 0:
            msg += f" Computed embeddings for {computed} samples."
        ctx.ops.notify(msg, variant="success")

    def label_positive(self, ctx: Any) -> None:
        """Label selected samples as positive."""
        session = self._get_session(ctx)
        if not session:
            ctx.ops.notify("No active session", variant="error")
            return

        selected = list(ctx.selected or [])
        if not selected:
            ctx.ops.notify("No samples selected", variant="warning")
            return

        session.add_positives(selected)
        self._save_session(ctx, session)
        self._ensure_training_label_fields(ctx, session)

        ctx.dataset.set_values(
            session.train_positive_field,
            {sid: True for sid in selected},
            key_field="id",
        )
        ctx.dataset.set_values(
            session.train_negative_field,
            {sid: None for sid in selected},
            key_field="id",
        )

        ctx.ops.notify(
            f"Added {len(selected)} samples as positive. "
            f"Total positives: {len(session.positive_ids)}",
            variant="success",
        )

    def label_negative(self, ctx: Any) -> None:
        """Label selected samples as negative."""
        session = self._get_session(ctx)
        if not session:
            ctx.ops.notify("No active session", variant="error")
            return

        selected = list(ctx.selected or [])
        if not selected:
            ctx.ops.notify("No samples selected", variant="warning")
            return

        session.add_negatives(selected)
        self._save_session(ctx, session)
        self._ensure_training_label_fields(ctx, session)

        ctx.dataset.set_values(
            session.train_negative_field,
            {sid: True for sid in selected},
            key_field="id",
        )
        ctx.dataset.set_values(
            session.train_positive_field,
            {sid: None for sid in selected},
            key_field="id",
        )

        ctx.ops.notify(
            f"Added {len(selected)} samples as negative. "
            f"Total negatives: {len(session.negative_ids)}",
            variant="success",
        )

    def train_and_apply(self, ctx: Any) -> None:
        """Train selected model on labels and apply to entire dataset using DataLoader."""
        session = self._get_session(ctx)
        if not session:
            ctx.ops.notify("No active session", variant="error")
            return

        if not session.can_train():
            ctx.ops.notify(
                "Need at least 1 positive and 1 negative sample to train",
                variant="warning",
            )
            return

        # Read current subset settings from UI (may have changed mid-session)
        ui_subset_size = getattr(ctx.panel.state, "working_subset_size", None)
        if ui_subset_size is not None:
            new_size = int(ui_subset_size)
            if new_size != session.working_subset_size:
                session.subset_ids = []
            session.working_subset_size = new_size
        ui_randomize = getattr(ctx.panel.state, "randomize_subset", None)
        if ui_randomize is not None:
            new_randomize = bool(ui_randomize)
            if new_randomize != session.randomize_subset:
                session.subset_ids = []
            session.randomize_subset = new_randomize

        # Read model settings from UI (may have changed mid-session)
        ui_model_name = getattr(ctx.panel.state, "model_name", None)
        if ui_model_name is not None:
            session.model_name = ui_model_name
        session.model_hyperparams = self._collect_hyperparams(
            ctx, session.model_name
        )

        ctx.ops.notify(f"Training {session.model_name}...", variant="info")

        # Create model from local few-shot model factory
        model = get_model(session.model_name, session.model_hyperparams)

        # Ensure embeddings for labeled samples and inference view
        zoo_model_name = EMBEDDING_ZOO_MODELS.get(
            session.embedding_model, "resnet18-imagenet-torch"
        )
        computed = 0

        labeled_ids = session.positive_ids + session.negative_ids
        if labeled_ids:
            labeled_view = ctx.dataset.select(labeled_ids)
            computed += self._ensure_embeddings(
                ctx, labeled_view, session.embedding_field, zoo_model_name
            )

        # Get inference view (possibly subset).
        # Materialize to ID-based select so the view is stable
        # (ctx.view may be lazily filtered on the prediction field
        # which we clear to None for in-scope samples before writing).
        inference_view, inference_count = self._get_inference_view(
            ctx, session
        )
        inference_view = ctx.dataset.select(inference_view.values("id"))
        computed += self._ensure_embeddings(
            ctx, inference_view, session.embedding_field, zoo_model_name
        )

        if computed > 0:
            ctx.ops.notify(
                f"Computed embeddings for {computed} new samples",
                variant="success",
            )

        # Train on labeled subset
        pos_view = ctx.dataset.select(session.positive_ids)
        neg_view = ctx.dataset.select(session.negative_ids)

        pos_emb_values = pos_view.values(session.embedding_field)
        neg_emb_values = neg_view.values(session.embedding_field)

        pos_emb = np.array(
            [emb for emb in pos_emb_values if emb is not None], dtype=np.float32
        )
        neg_emb = np.array(
            [emb for emb in neg_emb_values if emb is not None], dtype=np.float32
        )

        missing_pos = len(pos_emb_values) - len(pos_emb)
        missing_neg = len(neg_emb_values) - len(neg_emb)
        if missing_pos > 0 or missing_neg > 0:
            ctx.ops.notify(
                "Some labeled samples are missing embeddings and were skipped "
                f"(missing positives={missing_pos}, missing negatives={missing_neg})",
                variant="warning",
            )

        if len(pos_emb) == 0 or len(neg_emb) == 0:
            ctx.ops.notify(
                "Training requires at least 1 positive and 1 negative sample "
                "with valid embeddings",
                variant="error",
            )
            return

        embeddings = np.vstack([pos_emb, neg_emb]).astype(np.float32)
        labels = np.array([1] * len(pos_emb) + [0] * len(neg_emb))

        model.fit_step([{"embeddings": embeddings, "labels": labels}])
        train_count = len(embeddings)

        # Build output processor for LinearSVMModel (has one)
        if hasattr(model, "build_output_processor"):
            try:
                model.build_output_processor(classes=["negative", "positive"])
            except Exception:
                pass  # Model may not support output processor

        # Run inference using DataLoader on the inference view (possibly subset)
        get_item = EmbeddingsGetItem()
        get_item.field_mapping = {"embeddings": session.embedding_field}

        from torch.utils.data import DataLoader

        torch_dataset = inference_view.to_torch(
            get_item,
            vectorize=True,
            skip_failures=session.skip_failures,
        )
        dataloader = DataLoader(
            torch_dataset,
            batch_size=session.batch_size,
            num_workers=session.num_workers,
            collate_fn=collate_fn,
        )

        # Run inference and collect predictions
        predictions_map: dict[str, fo.Classification] = {}
        for batch in dataloader:
            # Get sample IDs before prediction
            sample_ids = batch["ids"]

            # Skip empty batches (can happen with skip_failures=True)
            if len(sample_ids) == 0:
                continue

            # Prepare batch for model (remove ids)
            model_batch = {
                "embeddings": batch["embeddings"].numpy()
            }

            # Run prediction
            output = model.predict(model_batch)

            # Extract probabilities (pass model_name for correct score handling)
            probs = extract_probability(output, session.model_name)

            # Convert to FiftyOne Classifications
            for sample_id, prob in zip(sample_ids, probs):
                label = "positive" if prob >= 0.5 else "negative"
                confidence = prob if label == "positive" else 1.0 - prob
                predictions_map[sample_id] = fo.Classification(
                    label=label,
                    confidence=float(confidence),
                )

        # Enforce user-provided training labels exactly on labeled IDs.
        # This avoids contradiction between explicit training labels and
        # displayed predictions for the same samples.
        for sample_id in session.positive_ids:
            predictions_map[sample_id] = fo.Classification(
                label="positive",
                confidence=1.0,
            )
        for sample_id in session.negative_ids:
            predictions_map[sample_id] = fo.Classification(
                label="negative",
                confidence=1.0,
            )

        # Write predictions cumulatively: update current inference outputs,
        # keep prior predictions for samples not scored this iteration.
        ctx.dataset.set_values(
            session.label_field,
            predictions_map,
            key_field="id",
        )

        # Write training label fields for sidebar visibility
        self._ensure_training_label_fields(ctx, session)
        ctx.dataset.set_values(
            session.train_positive_field,
            {sid: True for sid in session.positive_ids},
            key_field="id",
        )
        ctx.dataset.set_values(
            session.train_negative_field,
            {sid: True for sid in session.negative_ids},
            key_field="id",
        )

        session.iteration += 1
        self._save_session(ctx, session)

        subset_info = (
            f" (subset: {inference_count})"
            if session.working_subset_size > 0
            else ""
        )
        ctx.ops.notify(
            f"Iteration {session.iteration}: Trained on {train_count} samples, "
            f"labeled {inference_count} samples{subset_info}. "
            f"Predictions were written to '{session.label_field}'.",
            variant="success",
        )

    def view_predictions(self, ctx: Any) -> None:
        """Set App view to predicted samples sorted positive-first."""
        session = self._get_session(ctx)
        if not session:
            ctx.ops.notify("No active session", variant="error")
            return

        view = ctx.dataset.exists(session.label_field)
        if len(view) == 0:
            ctx.ops.notify(
                f"No predictions found in '{session.label_field}'",
                variant="warning",
            )
            return

        view = view.sort_by(f"{session.label_field}.label", reverse=True)
        ctx.ops.set_view(view)
        ctx.ops.notify("Switched to predictions view", variant="info")

    def reset_session(self, ctx: Any) -> None:
        """Clear session and delete prediction/training label fields."""
        session = self._get_session(ctx)
        if session:
            schema = ctx.dataset.get_field_schema()
            for field_name in (
                session.label_field,
                session.train_positive_field,
                session.train_negative_field,
            ):
                if field_name in schema:
                    ctx.dataset.delete_sample_field(field_name)

        self._clear_session(ctx)
        ctx.ops.notify("Session reset", variant="info")

    def render(self, ctx: Any) -> types.Property:
        """Render the panel UI based on active session state."""
        panel = types.Object()

        try:
            session = self._get_session(ctx)
        except Exception:
            session = None

        if session is None:
            panel.md(
                "## Few-Shot Learning\n\n"
                "Train a classifier by labeling positive and negative samples.\n\n"
                "1. Select embedding model and start session\n"
                "2. Label samples as positive/negative\n"
                "3. Select classifier model, train & label dataset\n"
                "4. Review predictions and iterate"
            )

            # Embedding setup
            panel.md("---\n**Embeddings:**")

            selected_emb_model = (
                getattr(ctx.panel.state, "embedding_model", None) or "ResNet18"
            )

            emb_model_dropdown = types.DropdownView()
            for display_name in EMBEDDING_ZOO_MODELS.keys():
                emb_model_dropdown.add_choice(display_name, label=display_name)
            panel.str(
                "embedding_model",
                label="Embedding Model",
                view=emb_model_dropdown,
                default="ResNet18",
                on_change=self.on_change_embedding_model,
            )

            panel.str(
                "embedding_field",
                label="Embedding Field",
                default=EMBEDDING_FIELD_DEFAULTS.get(
                    selected_emb_model, "resnet18_embeddings"
                ),
                description="Field name for embeddings "
                "(computed automatically if missing)",
            )

            # Advanced settings
            panel.md("---\n**Advanced Settings:**")

            panel.int(
                "batch_size",
                label="Batch Size",
                default=16,
                description="Batch size for inference DataLoader",
            )
            panel.int(
                "num_workers",
                label="Num Workers",
                default=8,
                description="Number of DataLoader workers "
                "(0 for main thread)",
            )
            panel.bool(
                "skip_failures",
                label="Skip Failures",
                default=True,
                description="Skip samples that fail to load",
            )

            # Subset sampling settings
            panel.md("---\n**Subset Sampling:**")

            panel.int(
                "working_subset_size",
                label="Working Subset Size",
                default=0,
                description="Limit inference to N samples "
                "(0 = no limit). "
                "Labeled samples are always included.",
            )
            panel.bool(
                "randomize_subset",
                label="Randomize Each Iteration",
                default=False,
                description="Re-sample a new subset each time "
                "'Train & Label Dataset' is clicked",
            )

            panel.btn(
                "start_session",
                label="Start Session",
                on_click=self.start_session,
                variant="contained",
            )
        else:
            stats = session.get_stats()
            panel.md(
                f"## Iteration {stats['iteration']}\n\n"
                f"**Positives:** {stats['positives']} | "
                f"**Negatives:** {stats['negatives']}"
            )
            panel.btn(
                "reset",
                label="Reset Session",
                on_click=self.reset_session,
                variant="outlined",
            )

            panel.md("---\n**Select samples in the grid, then label them:**")

            panel.btn(
                "label_pos",
                label=f"Label Positive ({stats['positives']})",
                on_click=self.label_positive,
                variant="contained",
            )
            panel.btn(
                "label_neg",
                label=f"Label Negative ({stats['negatives']})",
                on_click=self.label_negative,
                variant="outlined",
            )

            # Model selection
            panel.md("---\n**Model:**")
            model_dropdown = types.DropdownView()
            for m in SUPPORTED_MODELS:
                model_dropdown.add_choice(m, label=m)
            panel.str(
                "model_name",
                label="Model",
                view=model_dropdown,
                default=session.model_name,
            )

            selected_model = (
                getattr(ctx.panel.state, "model_name", None)
                or session.model_name
            )
            self._render_hyperparams(panel, selected_model)

            panel.md("---")
            panel.btn(
                "train",
                label="Train & Label Dataset",
                on_click=self.train_and_apply,
                variant="contained",
                disabled=not session.can_train(),
            )
            panel.btn(
                "view_predictions",
                label="View Predictions",
                on_click=self.view_predictions,
                variant="outlined",
            )

            # Subset sampling settings (visible during active session)
            panel.md("---\n**Subset Sampling:**")
            panel.int(
                "working_subset_size",
                label="Working Subset Size",
                default=session.working_subset_size,
                description="Limit inference to N samples " "(0 = no limit)",
            )
            panel.bool(
                "randomize_subset",
                label="Randomize Each Iteration",
                default=session.randomize_subset,
                description="Re-sample subset each iteration",
            )

        return types.Property(panel)
