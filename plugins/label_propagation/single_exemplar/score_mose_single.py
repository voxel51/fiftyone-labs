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

_PLUGINS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

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
    sequence_view: fo.DatasetView,
    label_idxs: List[int],
) -> fo.DatasetView:
    """
    Match ``test_mose`` ``partially_labeled_image_dataset_view`` for one sequence.

    Seeds ``labels_test`` on exemplar frames
    from ``ground_truth`` (same as test: only the first frame by default).
    """
    _ensure_detection_field(sequence_view._dataset, LABEL_FIELD)

    for ii, sample in enumerate(sequence_view):
        if ii in label_idxs:
            sample[LABEL_FIELD] = sample[GT_FIELD]
        else:
            sample[LABEL_FIELD] = fo.Detections(detections=[])
        sample.save()

    logger.info(
        "Prepared sequence %r: %d frames, seeded %s at indices %s",
        sequence_view.first()["sequence_id"],
        len(sequence_view),  # type: ignore[arg-type]
        LABEL_FIELD,
        label_idxs,
    )
    return sequence_view


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
            "propagate_bidirectionally": True,
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
        result.result.get("message") if result.result else result,  # type: ignore[attr-defined]
    )


def score_sequence(
    seq_view: fo.DatasetView, exemplar_indices: List[int]
) -> None:
    """
    Evaluate propagation vs. ground truth;
    Write scores to a log file.
    """
    scores = evaluate_detections(
        seq_view,
        pred_field=PROPAGATED_FIELD,
        gt_field=GT_FIELD,
    )

    exemplar_str = "_".join(str(idx) for idx in exemplar_indices)

    with open(
        f"scores_{seq_view.first()['sequence_id']}_{exemplar_str}.log", "w"
    ) as f:
        for i, score in enumerate(scores):
            f.write(f"{i},{score}\n")


def main():
    logger.setLevel(logging.INFO)
    args = _parse_args()

    logger.info("Loading MOSE-v2 zoo dataset (split=%s)...", MOSE_SPLIT)
    dataset = foz.load_zoo_dataset(MOSE_ZOO_URL, split=MOSE_SPLIT)

    sequence_ids = _resolve_sequence_ids(dataset)
    logger.info(
        "Processing %d sequence(s): %s", len(sequence_ids), sequence_ids
    )

    sequence_iter = sequence_ids
    if not args.no_progress:
        from tqdm import tqdm

        sequence_iter = tqdm(sequence_ids, desc="MOSE sequences")

    for sequence_id in sequence_iter:
        logger.info("=== Sequence %r ===", sequence_id)

        sequence_view = dataset.match(F("sequence_id") == sequence_id).sort_by(
            FRAME_SORT_FIELD
        )

        for label_idx in np.arange(0, len(sequence_view), step=args.step):
            sequence_view = prepare_sequence(sequence_view, [label_idx])
            propagate_sequence(dataset, sequence_view)
            score_sequence(sequence_view, [label_idx])


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
        "--step",
        type=int,
        default=5,
        help="Step size at which to try seeding exemplars (for the single-exemplar loop)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
