"""
Score SAM2 label propagation on the synthetic label-prop dataset.

Seeds ``ground_truth`` onto selected frame indices (``LABEL_IDXS``), runs
propagation per sequence, and writes per-sample scores vs. ``ground_truth``.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

_PLUGINS_DIR = Path(__file__).resolve().parent.parent
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

from label_propagation.propagation import (  # type: ignore
    add_detection_field_if_not_exists,
    delete_field_if_exists,
)
from label_propagation.suc_utils import evaluate_detections  # type: ignore


logger = logging.getLogger(__name__)

# Frame indices (within each sequence, sorted by ``frame_number``) used as seeds.
LABEL_IDXS: List[int] = [0]

DEFAULT_DATASET_NAME = "synthetic_label_prop"
PROPAGATION_METHOD = "sam2"
PROPAGATION_BATCH_SIZE = 256
SORT_FIELD = "frame_number"
GT_FIELD = "ground_truth"


def _seed_suffix(label_idxs: List[int]) -> str:
    return "_".join(str(i) for i in label_idxs)


def label_field_name(label_idxs: List[int]) -> str:
    return f"label_{_seed_suffix(label_idxs)}"


def propagated_field_name(label_idxs: List[int]) -> str:
    return f"{label_field_name(label_idxs)}_propagated"


def score_field_name(label_idxs: List[int]) -> str:
    return f"score_{_seed_suffix(label_idxs)}"


def _ensure_detection_field(dataset: fo.Dataset, field_name: str) -> None:
    if field_name not in dataset.get_field_schema():
        dataset.add_sample_field(
            field_name,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )


def setup_seed_labels(
    dataset: fo.Dataset,
    label_idxs: List[int],
    label_field: str,
) -> None:
    """Copy ``ground_truth`` onto seed frames; empty ``Detections`` elsewhere."""
    _ensure_detection_field(dataset, label_field)
    sequence_ids = dataset.distinct("sequence_id")

    for sequence_id in sequence_ids:
        seq_view = dataset.match(F("sequence_id") == sequence_id).sort_by(
            SORT_FIELD
        )
        samples = list(seq_view)
        n = len(samples)
        for idx in label_idxs:
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"LABEL_IDXS contains {idx}, but sequence {sequence_id!r} "
                    f"has only {n} frames (0..{n - 1})"
                )

        seq_view.set_values(
            label_field,
            [fo.Detections(detections=[]) for _ in range(n)],
        )
        for idx in label_idxs:
            sample = samples[idx]
            sample[label_field] = sample[GT_FIELD]
            sample.save()

        logger.info(
            "Seeded %s on sequence %r at frame indices %s",
            label_field,
            sequence_id,
            label_idxs,
        )


def propagate_sequence(
    dataset: fo.Dataset,
    sequence_id: str,
    label_field: str,
    propagated_field: str,
) -> None:
    """Run SAM2 propagation for one sequence via the plugin operator."""
    seq_view = dataset.match(F("sequence_id") == sequence_id).sort_by(
        SORT_FIELD
    )
    add_detection_field_if_not_exists(dataset, propagated_field)

    ctx = {
        "dataset": dataset,
        "view": seq_view,
        "params": {
            "input_annotation_field": label_field,
            "output_annotation_field": propagated_field,
            "propagation_method": PROPAGATION_METHOD,
            "sort_field": SORT_FIELD,
            "batch_size": PROPAGATION_BATCH_SIZE,
        },
    }
    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    logger.info(
        "Propagated sequence %r: %s",
        sequence_id,
        result.result.get("message") if result.result else result,
    )


def score_propagation(
    dataset: fo.Dataset,
    propagated_field: str,
    score_field: str,
) -> List[float]:
    """Evaluate propagated detections vs. ground truth; save per-sample scores."""
    if score_field not in dataset.get_field_schema():
        dataset.add_sample_field(score_field, fo.FloatField)

    all_scores: List[float] = []
    for sequence_id in sorted(dataset.distinct("sequence_id")):
        seq_view = dataset.match(F("sequence_id") == sequence_id).sort_by(
            SORT_FIELD
        )
        seq_scores = evaluate_detections(
            seq_view,
            pred_field=propagated_field,
            gt_field=GT_FIELD,
        )
        seq_view.set_values(score_field, seq_scores)
        all_scores.extend(seq_scores)
        logger.info(
            "Scored sequence %r: mean %s = %.4f",
            sequence_id,
            score_field,
            float(sum(seq_scores) / len(seq_scores)) if seq_scores else 0.0,
        )
    return all_scores


def score_synthetic_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    label_idxs: List[int] | None = None,
    reset_fields: bool = False,
) -> fo.Dataset:
    """Seed labels, propagate per sequence, and write scores on the dataset."""
    if not fo.dataset_exists(dataset_name):
        raise ValueError(f"Dataset {dataset_name!r} does not exist")

    label_idxs = list(LABEL_IDXS if label_idxs is None else label_idxs)
    if not label_idxs:
        raise ValueError("label_idxs must be non-empty")

    dataset = fo.load_dataset(dataset_name)
    if GT_FIELD not in dataset.get_field_schema():
        raise ValueError(f"Dataset missing {GT_FIELD!r}")

    label_field = label_field_name(label_idxs)
    propagated_field = propagated_field_name(label_idxs)
    score_field = score_field_name(label_idxs)

    if reset_fields:
        for field in (label_field, propagated_field, score_field):
            delete_field_if_exists(dataset, field)

    setup_seed_labels(dataset, label_idxs, label_field)

    for sequence_id in sorted(dataset.distinct("sequence_id")):
        propagate_sequence(dataset, sequence_id, label_field, propagated_field)

    scores = score_propagation(dataset, propagated_field, score_field)

    logger.info(
        "Wrote %d scores to %r (mean=%.4f)",
        len(scores),
        score_field,
        float(sum(scores) / len(scores)) if scores else 0.0,
    )
    return dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propagate and score labels on the synthetic dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"FiftyOne dataset name (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--reset-fields",
        action="store_true",
        help="Delete label / propagated / score fields before running",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    dataset = score_synthetic_dataset(
        dataset_name=args.dataset_name,
        reset_fields=args.reset_fields,
    )
    label_field = label_field_name(LABEL_IDXS)
    propagated_field = propagated_field_name(LABEL_IDXS)
    score_field = score_field_name(LABEL_IDXS)
    print(f"Dataset: {dataset.name}")
    print(f"  LABEL_IDXS: {LABEL_IDXS}")
    print(f"  label field: {label_field}")
    print(f"  propagated field: {propagated_field}")
    print(f"  score field: {score_field}")
    for sequence_id in sorted(dataset.distinct("sequence_id")):
        seq_view = dataset.match(F("sequence_id") == sequence_id).sort_by(
            SORT_FIELD
        )
        mean_score = sum(seq_view.values(score_field)) / len(seq_view)
        print(f"  {sequence_id}: mean {score_field} = {mean_score:.4f}")


if __name__ == "__main__":
    main()
