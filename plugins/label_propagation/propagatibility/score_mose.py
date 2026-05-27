"""
Score SAM2 propagation and compute backbone embeddings on MOSE-v2 sequences.

Mirrors ``tests/label_propagation/intensive/test_mose.py``: ``new_frame_number``,
``labels_test`` seeded on frame 0, propagate with ``sort_field=new_frame_number``,
then ``sam2_propagation_score`` and ``sam2_backbone_embeddings``.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz
import numpy as np
from fiftyone.core.expressions import ViewField as F

_PLUGINS_DIR = Path(__file__).resolve().parent.parent
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

from label_propagation.embedding_utils import get_sam2_embeddings  # type: ignore
from label_propagation.propagation import add_detection_field_if_not_exists  # type: ignore
from label_propagation.suc_utils import evaluate_detections  # type: ignore


logger = logging.getLogger(__name__)

# Frame indices (within each sequence, sorted by ``frame_number``) used as seeds.
LABEL_IDXS: List[int] = [0]

# ``None`` = all sequences; or indices into ``sorted(dataset.distinct("sequence_id"))``
# e.g. ``[6, 10]`` matches ``test_mose`` parametrized cases.
SEQUENCE_ID_INDICES: Optional[List[int]] = [6, 10]

MOSE_ZOO_URL = "https://github.com/voxel51/mose-v2"
MOSE_SPLIT = "train"
PROPAGATION_METHOD = "sam2"
PROPAGATION_BATCH_SIZE = 256
FRAME_SORT_FIELD = "frame_number"
GT_FIELD = "ground_truth"
LABEL_FIELD = "labels_test"
PROPAGATED_FIELD = "labels_test_propagated"
SCORE_FIELD = "sam2_propagation_score"
EMBEDDING_FIELD = "sam2_backbone_embeddings"
MIN_MEAN_SCORE = 0.4


def _resolve_sequence_ids(dataset: fo.Dataset) -> List[str]:
    all_ids = sorted(dataset.distinct("sequence_id"))  # type: ignore[arg-type]
    if SEQUENCE_ID_INDICES is None:
        return all_ids
    return [all_ids[i] for i in SEQUENCE_ID_INDICES]


def _ensure_detection_field(dataset: fo.Dataset, field_name: str) -> None:
    if field_name not in dataset.get_field_schema():
        dataset.add_sample_field(
            field_name,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )


def prepare_sequence(
    dataset: fo.Dataset,
    sequence_id: str,
    label_idxs: List[int],
) -> fo.DatasetView:
    """
    Match ``test_mose`` ``partially_labeled_image_dataset_view`` for one sequence.

    Sets ``new_frame_number`` to ``0..n-1``, seeds ``labels_test`` on exemplar frames
    from ``ground_truth`` (same as test: only the first frame by default).
    """
    _ensure_detection_field(dataset, LABEL_FIELD)
    seq_view = dataset.match(F("sequence_id") == sequence_id).sort_by(
        FRAME_SORT_FIELD
    )
    n = len(seq_view)
    for idx in label_idxs:
        if idx < 0 or idx >= n:
            raise ValueError(
                f"LABEL_IDXS contains {idx}, but sequence {sequence_id!r} "
                f"has only {n} frames (0..{n - 1})"
            )

    for idx in label_idxs:
        exemplar = seq_view.skip(idx).first()
        exemplar[LABEL_FIELD] = exemplar[GT_FIELD]
        exemplar.save()

    logger.info(
        "Prepared sequence %r: %d frames, seeded %s at indices %s",
        sequence_id,
        n,
        LABEL_FIELD,
        label_idxs,
    )
    return seq_view


def propagate_sequence(dataset: fo.Dataset, seq_view: fo.DatasetView) -> None:
    """Run SAM2 propagation (same operator ctx as ``test_propagate_labels_image``)."""
    add_detection_field_if_not_exists(dataset, PROPAGATED_FIELD)

    ctx = {
        "dataset": dataset,
        "view": seq_view,
        "params": {
            "input_annotation_field": LABEL_FIELD,
            "output_annotation_field": PROPAGATED_FIELD,
            "propagation_method": PROPAGATION_METHOD,
            "sort_field": FRAME_SORT_FIELD,
            "batch_size": PROPAGATION_BATCH_SIZE,
        },
    }
    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    logger.info(
        "Propagated %r: %s",
        seq_view.first()["sequence_id"],
        result.result.get("message") if result.result else result,
    )


def score_sequence(seq_view: fo.DatasetView) -> List[float]:
    """Evaluate propagation vs. ground truth; write ``sam2_propagation_score``."""
    dataset = seq_view._dataset
    if SCORE_FIELD not in dataset.get_field_schema():
        dataset.add_sample_field(SCORE_FIELD, fo.FloatField)

    scores = evaluate_detections(
        seq_view,
        pred_field=PROPAGATED_FIELD,
        gt_field=GT_FIELD,
    )
    seq_view.set_values(SCORE_FIELD, scores)
    mean_score = float(np.mean(scores)) if scores else 0.0
    logger.info(
        "Scored sequence %r: mean %s = %.4f (%d frames)",
        seq_view.first()["sequence_id"],
        SCORE_FIELD,
        mean_score,
        len(scores),
    )
    return scores


def compute_embeddings_sequence(seq_view: fo.DatasetView) -> None:
    """Compute and store ``sam2_backbone_embeddings`` for one sequence."""
    sequence_id = seq_view.first()["sequence_id"]
    logger.info(
        "Computing %s for sequence %r (%d samples)...",
        EMBEDDING_FIELD,
        sequence_id,
        len(seq_view),
    )
    get_sam2_embeddings(seq_view)
    logger.info("Finished %s for sequence %r", EMBEDDING_FIELD, sequence_id)


def score_mose_dataset(
    dataset: Optional[fo.Dataset] = None,
    label_idxs: Optional[List[int]] = None,
    show_progress: bool = True,
    skip_embeddings: bool = False,
) -> fo.Dataset:
    """Run full MOSE scoring pipeline per sequence."""
    label_idxs = list(LABEL_IDXS if label_idxs is None else label_idxs)
    if not label_idxs:
        raise ValueError("label_idxs must be non-empty")

    if dataset is None:
        logger.info("Loading MOSE-v2 zoo dataset (split=%s)...", MOSE_SPLIT)
        dataset = foz.load_zoo_dataset(MOSE_ZOO_URL, split=MOSE_SPLIT)

    if GT_FIELD not in dataset.get_field_schema():
        raise ValueError(f"Dataset missing {GT_FIELD!r}")

    sequence_ids = _resolve_sequence_ids(dataset)
    logger.info(
        "Processing %d sequence(s): %s", len(sequence_ids), sequence_ids
    )

    sequence_iter = sequence_ids
    if show_progress:
        from tqdm import tqdm

        sequence_iter = tqdm(sequence_ids, desc="MOSE sequences")

    for sequence_id in sequence_iter:
        logger.info("=== Sequence %r ===", sequence_id)
        seq_view = prepare_sequence(dataset, sequence_id, label_idxs)
        propagate_sequence(dataset, seq_view)
        scores = score_sequence(seq_view)
        if float(np.mean(scores)) <= MIN_MEAN_SCORE:
            raise RuntimeError(
                f"Sequence {sequence_id!r}: mean {SCORE_FIELD} = "
                f"{float(np.mean(scores)):.4f} (expected > {MIN_MEAN_SCORE})"
            )
        if not skip_embeddings:
            compute_embeddings_sequence(seq_view)

    return dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propagate and score MOSE-v2 sequences.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars on the sequence loop",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip sam2_backbone_embeddings (faster; propagation + score only)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    dataset = score_mose_dataset(
        show_progress=not args.no_progress,
        skip_embeddings=args.skip_embeddings,
    )

    print(f"Dataset: {dataset.name}")
    print(f"  LABEL_IDXS: {LABEL_IDXS}")
    print(f"  sequences: {_resolve_sequence_ids(dataset)}")
    print(f"  propagation sort_field: {FRAME_SORT_FIELD}")

    for sequence_id in _resolve_sequence_ids(dataset):
        seq_view = dataset.match(F("sequence_id") == sequence_id).sort_by(
            FRAME_SORT_FIELD
        )
        scores = seq_view.values(SCORE_FIELD)
        mean_score = float(np.mean(scores))
        print(f"  {sequence_id}: mean {SCORE_FIELD} = {mean_score:.4f}")
        if mean_score <= MIN_MEAN_SCORE:
            raise SystemExit(
                f"FAIL {sequence_id}: mean {mean_score:.4f} <= {MIN_MEAN_SCORE}"
            )


if __name__ == "__main__":
    main()
