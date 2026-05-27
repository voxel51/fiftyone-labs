"""
Inspect SAM2 backbone embedding statistics on MOSE-v2 sequences.

Reuses plotting helpers from ``inspect_synthetic_dataset``. Expects
``sam2_backbone_embeddings`` to already exist (see ``score_mose.py``).

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
import fiftyone.zoo as foz

_PLUGINS_DIR = Path(__file__).resolve().parent.parent
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

from label_propagation.inspect_synthetic_dataset import (  # type: ignore
    EMBEDDING_FIELD,
    inspect_dataset,
)

logger = logging.getLogger(__name__)

MOSE_ZOO_URL = "https://github.com/voxel51/mose-v2"
MOSE_SPLIT = "train"
STATS_DIR_NAME = "mose_embedding_stats"

# ``None`` = all sequences; or indices into ``sorted(dataset.distinct("sequence_id"))``
SEQUENCE_ID_INDICES: Optional[List[int]] = [6, 10]


def _stats_dir(script_dir: Path) -> Path:
    return script_dir / STATS_DIR_NAME


def _resolve_sequence_ids(dataset: fo.Dataset) -> List[str]:
    all_ids = sorted(dataset.distinct("sequence_id"))  # type: ignore[arg-type]
    if SEQUENCE_ID_INDICES is None:
        return all_ids
    return [all_ids[i] for i in SEQUENCE_ID_INDICES]


def inspect_mose_dataset(
    dataset: Optional[fo.Dataset] = None,
    output_dir: Optional[Path] = None,
    show_progress: bool = True,
) -> Path:
    """Load MOSE (or use provided dataset) and save embedding stat plots."""
    if dataset is None:
        logger.info("Loading MOSE-v2 zoo dataset (split=%s)...", MOSE_SPLIT)
        dataset = foz.load_zoo_dataset(MOSE_ZOO_URL, split=MOSE_SPLIT)

    if EMBEDDING_FIELD not in dataset.get_field_schema():
        raise ValueError(
            f"Dataset has no {EMBEDDING_FIELD!r}; run score_mose.py first"
        )

    script_dir = Path(__file__).resolve().parent
    out_root = output_dir or _stats_dir(script_dir)
    sequence_ids = _resolve_sequence_ids(dataset)
    logger.info(
        "Inspecting %d sequence(s): %s (output: %s)",
        len(sequence_ids),
        sequence_ids,
        out_root,
    )

    return inspect_dataset(
        dataset,
        sequence_ids,
        out_root,
        show_progress=show_progress,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SAM2 embedding statistics for MOSE-v2 sequences.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output folder (default: <plugin>/{STATS_DIR_NAME}/)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    out = args.output_dir
    output_dir = Path(out) if out is not None else None
    saved = inspect_mose_dataset(
        output_dir=output_dir,
        show_progress=not args.no_progress,
    )
    print(f"Plots saved to {saved}")


if __name__ == "__main__":
    main()
