"""
Compare SAM2 propagation when seeding N exemplar frames via different selection methods.

Mirrors ``single_exemplar/score_mose_single.py``: load MOSE-v2, seed ``labels_test`` on
chosen frames, bidirectional propagation, per-frame scores logged to
``scores_{sequence_id}_{method}.log``.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence
import csv

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz
import numpy as np
from fiftyone.core.expressions import ViewField as F

_PLUGINS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

from label_propagation.embedding_utils import (  # type: ignore
    consecutive_asymmetric_hausdorff_max,
    get_sam2_embeddings,
    cycle_consistency_error,
    many_one_collapse_score,
    local_topology_distortion,
    pairwise_metric,
)
from label_propagation.propagation import add_detection_field_if_not_exists  # type: ignore
from label_propagation.suc_utils import evaluate_detections  # type: ignore


consecutive_cycle_consistency_error = pairwise_metric(cycle_consistency_error)
consecutive_many_one_collapse_score = pairwise_metric(many_one_collapse_score)
consecutive_local_topology_distortion = pairwise_metric(local_topology_distortion)

logger = logging.getLogger(__name__)

# ``None`` = all sequences; or indices into ``sorted(dataset.distinct("sequence_id"))``
SEQUENCE_ID_INDICES: Optional[List[int]] = [10]

MOSE_ZOO_URL = "https://github.com/voxel51/mose-v2"
MOSE_SPLIT = "train"
PROPAGATION_METHOD = "sam2"
PROPAGATION_BATCH_SIZE = 256
PROPAGATE_BIDIRECTIONALLY = False
FRAME_SORT_FIELD = "frame_number"
GT_FIELD = "ground_truth"
LABEL_FIELD = "labels_test"
PROPAGATED_FIELD = "labels_test_propagated"
EMBEDDING_FIELD = "sam2_backbone_embeddings"
CSV_FILE_PREFIX = "scores_fwd_"


EXEMPLAR_METHODS: List[str] = [
    "equally_spaced",
    "random_1",
    "random_2",
    "random_3",
    "hausdorff_delta",
    "cycle_consistency",
    "many_one_collapse",
    "local_topology_distortion",
    # baselines
    "first_frame_only",
    "alternate_frames",
    "custom"
]


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


def _load_sequence_embeddings(seq_view: fo.DatasetView) -> np.ndarray:
    """Stack ``sam2_backbone_embeddings`` in frame order, shape ``(N, D, H, W)``."""
    embeddings: List[np.ndarray] = []
    for sample in seq_view:
        emb = sample[EMBEDDING_FIELD]
        if emb is None:
            raise ValueError(
                f"Missing {EMBEDDING_FIELD!r} on sample {sample.id} "
                f"(sequence_id={sample['sequence_id']!r})"
            )
        embeddings.append(np.asarray(emb))
    return np.stack(embeddings, axis=0)


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


def select_exemplar_indices(
    method: str,
    n_frames: int,
    embeddings: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Return ``N_EXEMPLARS`` frame indices (0-based, sorted) for the given method.

    ``hausdorff_delta`` requires ``embeddings`` of shape ``(N, D, H, W)``.
    """
    if method == "first_frame_only":
        return [0]
    elif method == "alternate_frames":
        return list(range(0, n_frames, 2))
    elif method == "custom":
        # return [0, 9, 18]  # seq 6
        return [0, 12, 21, 31, 37, 45, 53, 69]  # seq 10

    N_EXEMPLARS = max(1, n_frames // 10)

    if method == "equally_spaced":
        if PROPAGATE_BIDIRECTIONALLY:
            idxs = np.linspace(0, n_frames - 1, N_EXEMPLARS, dtype=int)
            return sorted(int(i) for i in np.unique(idxs))
        else:
            idxs = np.linspace(0, n_frames, N_EXEMPLARS + 1, dtype=int)[:-1]
            return sorted([int(i) for i in np.unique(idxs)])

    elif method.startswith("random_"):
        seed = int(method.split("_", 1)[1])
        rng = np.random.default_rng(seed)
        if PROPAGATE_BIDIRECTIONALLY:
            chosen = rng.choice(n_frames, size=N_EXEMPLARS, replace=False)
            return sorted(int(i) for i in chosen)
        else:
            chosen = rng.choice(np.arange(1, n_frames), size=N_EXEMPLARS-1, replace=False)
            return [0] + sorted(int(i) for i in chosen)

    elif method == "hausdorff_delta":
        if embeddings is None:
            raise ValueError("hausdorff_delta requires precomputed embeddings")
        
        # Max patchwise diff between frame i and i-1; exclude frame 0.
        deltas = consecutive_cycle_consistency_error(
            embeddings,
            # show_progress=False,
        )
        print("\n\nconsecutive cycle consistency error deltas")
        print(deltas)

    elif method == "cycle_consistency":
        if embeddings is None:
            raise ValueError("cycle_consistency requires precomputed embeddings")
        
        # Max patchwise diff between frame i and i-1; exclude frame 0.
        deltas = consecutive_cycle_consistency_error(
            embeddings,
            # show_progress=False,
        )
        print("\n\nconsecutive cycle consistency error deltas")
        print(deltas)
        
    elif method == "many_one_collapse":
        if embeddings is None:
            raise ValueError("many_one_collapse requires precomputed embeddings")
        
        deltas = consecutive_many_one_collapse_score(
            embeddings,
            # show_progress=False,
        )
        print("\n\nconsecutive many one collapse score deltas")
        print(deltas)
    
    elif method == "local_topology_distortion":
        if embeddings is None:
            raise ValueError("local_topology_distortion requires precomputed embeddings")
        
        deltas = consecutive_local_topology_distortion(
            embeddings,
            # show_progress=False,
        )
        print("\n\nconsecutive local topology distortion deltas")
        print(deltas)
        
    else:
        raise ValueError(f"Unknown exemplar method: {method!r}")
    
    frame_indices = np.arange(1, n_frames)
    if PROPAGATE_BIDIRECTIONALLY:
        top = np.argpartition(-deltas, N_EXEMPLARS - 1)[:N_EXEMPLARS]
        return sorted(int(frame_indices[i]) for i in top)
    else:
        top = np.argpartition(-deltas, N_EXEMPLARS - 1)[:N_EXEMPLARS-1]
        return [0] + sorted(int(frame_indices[i]) for i in top)


def prepare_sequence(
    sequence_view: fo.DatasetView,
    label_idxs: Sequence[int],
) -> fo.DatasetView:
    """Seed ``labels_test`` on exemplar frames from ``ground_truth``; clear others."""
    _ensure_detection_field(sequence_view._dataset, LABEL_FIELD)
    label_set = set(label_idxs)

    for ii, sample in enumerate(sequence_view):
        if ii in label_set:
            sample[LABEL_FIELD] = sample[GT_FIELD]
        else:
            sample[LABEL_FIELD] = fo.Detections(detections=[])
        sample.save()

    logger.info(
        "Prepared sequence %r: %d frames, seeded %s at indices %s",
        sequence_view.first()["sequence_id"],
        len(sequence_view),  # type: ignore[arg-type]
        LABEL_FIELD,
        sorted(label_idxs),
    )
    return sequence_view


def propagate_sequence(dataset: fo.Dataset, seq_view: fo.DatasetView) -> None:
    """Run SAM2 propagation with bidirectional tracking."""
    add_detection_field_if_not_exists(dataset, PROPAGATED_FIELD)

    ctx = {
        "dataset": dataset,
        "view": seq_view,
        "params": {
            "input_annotation_field": LABEL_FIELD,
            "output_annotation_field": PROPAGATED_FIELD,
            "propagation_method": PROPAGATION_METHOD,
            "propagate_bidirectionally": PROPAGATE_BIDIRECTIONALLY,
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


def score_sequence(seq_view: fo.DatasetView, method: str) -> None:
    """Evaluate propagation vs. ground truth; write per-frame scores to a log file."""
    scores = evaluate_detections(
        seq_view,
        pred_field=PROPAGATED_FIELD,
        gt_field=GT_FIELD,
    )

    sequence_id = seq_view.first()["sequence_id"]

    # # write to .log file
    # log_path = Path(f"scores_{sequence_id}_{method}.log")
    # with log_path.open("w") as f:
    #     for i, score in enumerate(scores):
    #         f.write(f"{i},{score}\n")
    # logger.info("Wrote %s (%d frames)", log_path, len(scores))
    
    # write to .csv file
    csv_path = Path(f"{CSV_FILE_PREFIX}{sequence_id}.csv")
    with csv_path.open("a") as f:
        writer = csv.writer(f)
        writer.writerow([method, *scores])
    logger.info("Appended to %s", csv_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    logger.info("Loading MOSE-v2 zoo dataset (split=%s)...", MOSE_SPLIT)
    dataset = foz.load_zoo_dataset(MOSE_ZOO_URL, split=MOSE_SPLIT)

    sequence_ids = _resolve_sequence_ids(dataset)
    logger.info(
        "Processing %d sequence(s): %s", len(sequence_ids), sequence_ids
    )

    methods = EXEMPLAR_METHODS
    if args.methods:
        methods = list(args.methods)

    sequence_iter: Sequence[str] = sequence_ids
    if not args.no_progress:
        from tqdm import tqdm

        sequence_iter = tqdm(sequence_ids, desc="MOSE sequences")

    for sequence_id in sequence_iter:
        logger.info("=== Sequence %r ===", sequence_id)

        sequence_view = dataset.match(F("sequence_id") == sequence_id).sort_by(
            FRAME_SORT_FIELD
        )
        n_frames = len(sequence_view)

        embeddings: Optional[np.ndarray] = None
        
        if any(method in [
            "hausdorff_delta", "cycle_consistency", "many_one_collapse", "local_topology_distortion"
        ] for method in methods):
            if not args.skip_embeddings:
                compute_embeddings_sequence(sequence_view)
            embeddings = _load_sequence_embeddings(sequence_view)

        method_iter = methods
        if not args.no_progress:
            from tqdm import tqdm

            method_iter = tqdm(methods, desc=f"{sequence_id} methods", leave=False)

        for method in method_iter:
            label_idxs = select_exemplar_indices(
                method,
                n_frames,
                embeddings=embeddings,
            )
            logger.info(
                "\n\n=== Sequence %r method %r -> exemplar indices %s ===\n",
                sequence_id,
                method,
                label_idxs,
            )
            prepare_sequence(sequence_view, label_idxs)
            propagate_sequence(dataset, sequence_view)
            score_sequence(sequence_view, f"{method}_{len(label_idxs)}ex")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare N-exemplar SAM2 propagation on MOSE-v2 using different "
            "exemplar selection strategies."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help=(
            "Do not compute sam2_backbone_embeddings (hausdorff_delta requires "
            "them to already exist on the dataset)"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=EXEMPLAR_METHODS,
        metavar="METHOD",
        help=f"Subset of methods to run (default: all {EXEMPLAR_METHODS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
