"""
Plot frame-wise forward-propagation scores from N-exemplar selection experiments.

Reads ``scores_fwd_{sequence_id}.csv`` files (one row per exemplar method, columns
are frame indices) and draws one vertically stacked subplot per sequence.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent

# --- Configuration -----------------------------------------------------------

SEQUENCES: List[str] = [
    "00e92ab4",
    "01e3cec8",
]

CSV_FILE_PREFIX = "scores_fwd_"

# Subset of methods from compare_mose.py::EXEMPLAR_METHODS.

METHODS_CONFIG: Dict[str, Dict[str, str | float]] = {
    "first_frame_only": {"color": "black", "linestyle": "-.", "alpha": 1.0},
    "alternate_frames": {"color": "black", "linestyle": "--", "alpha": 1.0},
    # "random_1 (10%)": {"color": "crimson", "linestyle": "-", "alpha": 0.5},
    # "random_2 (10%)": {"color": "mediumvioletred", "linestyle": "-", "alpha": 0.5},
    # "random_3 (10%)": {"color": "hotpink", "linestyle": "-", "alpha": 0.5},
    "equally_spaced (10%)": {"color": "dodgerblue", "linestyle": "-", "alpha": 1.0},
    "hausdorff_delta (10%)": {"color": "forestgreen", "linestyle": "-", "alpha": 1.0},
    "cycle_consistency (10%)": {"color": "darkorange", "linestyle": "-", "alpha": 1.0},
    # "local_topology_distortion (10%)": {"color": "gold", "linestyle": "-", "alpha": 1.0},
    # "many_one_collapse (10%)": {"color": "mediumslateblue", "linestyle": "-", "alpha": 1.0},
    # "custom (10%)": {"color": "tomato", "linestyle": ":", "alpha": 1.0},
}

OUTPUT_PATH: Optional[Path] = _SCRIPT_DIR / "scores_fwdprop_plot.png"
FIG_WIDTH = 12.0
FIG_HEIGHT_PER_SEQUENCE = 3.0
Y_LABEL = "Instance-wise Mean IoU"
X_LABEL = "Frame #"
TITLE_TEMPLATE = "Sequence {sequence_id}"
SHOW_LEGEND = True

# -----------------------------------------------------------------------------


def _csv_path(sequence_id: str) -> Path:
    return _SCRIPT_DIR / f"{CSV_FILE_PREFIX}{sequence_id}.csv"


def _load_scores_csv(path: Path) -> Dict[str, Tuple[List[int], List[float]]]:
    """Return ``{exemplar_row_name: (frame_indices, scores)}``."""
    rows: Dict[str, Tuple[List[int], List[float]]] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        frame_cols = [int(col) for col in header[1:]]

        for row in reader:
            if not row:
                continue
            exemplar = row[0]
            scores: List[float] = []
            frames: List[int] = []
            for frame_idx, raw in zip(frame_cols, row[1:]):
                if raw == "":
                    continue
                scores.append(float(raw))
                frames.append(frame_idx)
            rows[exemplar] = (frames, scores)

    return rows


def _find_exemplar_row(
    rows: Dict[str, Tuple[List[int], List[float]]],
    method: str,
) -> Optional[str]:
    method_type = method.split(" ")[0]
    prefix = f"{method_type}_"
    matches = [name for name in rows if name.startswith(prefix)]
    if not matches:
        return None
    if len(matches) > 1:
        matches.sort()
        logger.warning(
            "Multiple rows match method %r: %s; using %r",
            method_type,
            matches,
            matches[0],
        )
    return matches[0]


def plot_scores_fwd() -> Path:
    n_sequences = len(SEQUENCES)
    if n_sequences == 0:
        raise ValueError("SEQUENCES must contain at least one sequence id")

    fig, axes = plt.subplots(
        n_sequences,
        1,
        figsize=(FIG_WIDTH, FIG_HEIGHT_PER_SEQUENCE * n_sequences),
        sharex=False,
        squeeze=False,
    )

    legend_handles: Dict[str, plt.Line2D] = {}

    for ax_idx, sequence_id in enumerate(SEQUENCES):
        ax = axes[ax_idx, 0]
        csv_path = _csv_path(sequence_id)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing scores CSV: {csv_path}")

        rows = _load_scores_csv(csv_path)
        plotted_methods: List[str] = []

        for method in METHODS_CONFIG.keys():
            exemplar_row = _find_exemplar_row(rows, method)
            if exemplar_row is None:
                logger.info(
                    "Skipping method %r for sequence %r (no matching row)",
                    method,
                    sequence_id,
                )
                continue

            frames, scores = rows[exemplar_row]
            config = METHODS_CONFIG.get(method, {})
            color = config.get("color", None)
            linestyle = config.get("linestyle", "-")
            alpha = config.get("alpha", 1.0)
            label = method

            (line,) = ax.plot(
                frames,
                scores,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                alpha=alpha,
                label=label,
            )
            legend_handles.setdefault(method, line)
            plotted_methods.append(method)

        ax.set_title(TITLE_TEMPLATE.format(sequence_id=sequence_id))
        ax.set_ylabel(Y_LABEL)
        ax.set_xlim(left=0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

        if not plotted_methods:
            logger.warning("No methods plotted for sequence %r", sequence_id)

        if ax_idx == n_sequences - 1:
            ax.set_xlabel(X_LABEL)

    if SHOW_LEGEND and legend_handles:
        fig.legend(
            [legend_handles[m] for m in METHODS_CONFIG.keys() if m in legend_handles],
            [m for m in METHODS_CONFIG.keys() if m in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(5, len(legend_handles)),
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout()

    if OUTPUT_PATH is None:
        raise ValueError("OUTPUT_PATH is None; set a path to save the figure")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", OUTPUT_PATH)
    return OUTPUT_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    plot_scores_fwd()


if __name__ == "__main__":
    main()
