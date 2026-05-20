"""
Inspect SAM2 backbone embedding deltas on the synthetic label-prop dataset.

Loads ``sam2_backbone_embeddings`` per sequence, computes consecutive-frame
element-wise diffs, and saves statistic plots (no interactive display).

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import fiftyone as fo
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)

EMBEDDING_FIELD = "sam2_backbone_embeddings"
DEFAULT_DATASET_NAME = "synthetic_label_prop"
STATS_DIR_NAME = "synthetic_label_prop_embedding_stats"

STAT_NAMES = ("min", "max", "mean", "median", "std")
STAT_FNS = {
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
}

# LaTeX for plot labels (D×H×W backbone embeddings; Δx = x_{i+1} - x_i).
NOTATION_CAPTION = (
    r"Adjacent embeddings $x_i, x_{i+1} \in \mathbb{R}^{D \times H \times W}$;"
    r" $\Delta x = x_{i+1} - x_i$"
)

AGGREGATE_FORMULAS = {
    "min": r"\min_{d,h,w}(\Delta x)_{dhw}",
    "max": r"\max_{d,h,w}(\Delta x)_{dhw}",
    "mean": r"\frac{1}{DHW}\sum_{d,h,w}(\Delta x)_{dhw}",
    "median": r"\mathrm{median}_{d,h,w}(\Delta x)_{dhw}",
    "std": r"\mathrm{std}_{d,h,w}(\Delta x)_{dhw}",
}

PER_CHANNEL_FORMULAS = {
    "min": r"\min_{h,w}(\Delta x)_{dhw}",
    "max": r"\max_{h,w}(\Delta x)_{dhw}",
    "mean": r"\frac{1}{HW}\sum_{h,w}(\Delta x)_{dhw}",
    "median": r"\mathrm{median}_{h,w}(\Delta x)_{dhw}",
    "std": r"\mathrm{std}_{h,w}(\Delta x)_{dhw}",
}

L2_GLOBAL_FORMULA = r"\|\Delta x\|_2 = \sqrt{\sum_{d,h,w}(\Delta x)_{dhw}^2}"

L2_PER_CHANNEL_FORMULA = r"\|\Delta x\|_{2,d} = \sqrt{\sum_{h,w}(\Delta x)_{dhw}^2}"

COSINE_DISTANCE_FORMULA = (
    r"1 - \frac{\sum_{d,h,w} x_{i,dhw}\, x_{i+1,dhw}}"
    r"{\|x_i\|_2\,\|x_{i+1}\|_2}"
)

MEAN_ABS_SPATIAL_FORMULA = r"\frac{1}{D}\sum_{d}\left|(\Delta x)_{dhw}\right|"

FRACTION_ABOVE_FORMULA = (
    r"\frac{1}{DHW}\sum_{d,h,w}\mathbf{1}\!\left[\left|(\Delta x)_{dhw}\right|>\tau\right]"
)

X_LABEL = r"frame number $j$ (later frame in pair $x_{j-1}, x_j$)"


def _add_notation_caption(fig, y: float = 0.98) -> None:
    fig.text(0.5, y, NOTATION_CAPTION, ha="center", va="top", fontsize=9)


def _set_ylabel_with_formula(ax, formula: str, short: str = "") -> None:
    """Set y-axis label with a LaTeX formula on the line below."""
    if short:
        ax.set_ylabel(f"{short}\n${formula}$", fontsize=10)
    else:
        ax.set_ylabel(f"${formula}$", fontsize=10)


def _legend_formula(stat: str, scope: str) -> str:
    if scope == "aggregate":
        return f"${AGGREGATE_FORMULAS[stat]}$"
    if scope == "per_channel":
        return f"${PER_CHANNEL_FORMULAS[stat]}$"
    raise ValueError(f"Unknown scope: {scope}")


def _stats_dir(script_dir: Path) -> Path:
    return script_dir / STATS_DIR_NAME


def _load_sequence_embeddings(
    dataset: fo.Dataset, sequence_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (frame_numbers, embeddings) with shape (N,) and (N, C, H, W)."""
    view = (
        dataset.match({"sequence_id": sequence_id})
        .sort_by("frame_number")
        .select_fields(["frame_number", EMBEDDING_FIELD])
    )
    frame_numbers: List[int] = []
    embeddings: List[np.ndarray] = []
    for sample in view:
        emb = sample[EMBEDDING_FIELD]
        if emb is None:
            raise ValueError(
                f"Missing {EMBEDDING_FIELD!r} on sample {sample.id} "
                f"(sequence_id={sequence_id!r})"
            )
        frame_numbers.append(int(sample["frame_number"]))
        embeddings.append(np.asarray(emb))

    return np.asarray(frame_numbers), np.stack(embeddings, axis=0)


def _consecutive_diffs(embeddings: np.ndarray) -> np.ndarray:
    """Element-wise diffs between consecutive frames, shape (N-1, C, H, W)."""
    return embeddings[1:] - embeddings[:-1]


def _diff_x_axis(frame_numbers: np.ndarray) -> np.ndarray:
    """X-axis for diff plots: frame number of the later frame in each pair."""
    return frame_numbers[1:]


def _aggregate_stat_curve(diffs: np.ndarray, stat: str) -> np.ndarray:
    """Scalar stat over C,H,W for each diff step -> length N-1."""
    flat = diffs.reshape(diffs.shape[0], -1)
    return STAT_FNS[stat](flat, axis=1)


def _per_channel_stat_curves(diffs: np.ndarray, stat: str) -> np.ndarray:
    """Per-channel stat over H,W -> shape (N-1, C)."""
    return STAT_FNS[stat](diffs, axis=(2, 3))


def _l2_norm_curve(diffs: np.ndarray) -> np.ndarray:
    flat = diffs.reshape(diffs.shape[0], -1)
    return np.linalg.norm(flat, axis=1)


def _cosine_distance_curve(embeddings: np.ndarray) -> np.ndarray:
    a = embeddings[:-1].reshape(embeddings.shape[0] - 1, -1)
    b = embeddings[1:].reshape(embeddings.shape[0] - 1, -1)
    dot = np.sum(a * b, axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.maximum(na * nb, 1e-12)
    return 1.0 - dot / denom


def _mean_abs_spatial_maps(diffs: np.ndarray) -> np.ndarray:
    """Mean |diff| over channels -> shape (N-1, H, W)."""
    return np.mean(np.abs(diffs), axis=1)


def plot_aggregate_stats(
    sequence_id: str,
    x: np.ndarray,
    diffs: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for stat in STAT_NAMES:
        y = _aggregate_stat_curve(diffs, stat)
        ax.plot(
            x,
            y,
            marker="o",
            label=_legend_formula(stat, "aggregate"),
            linewidth=2,
        )
    ax.set_xlabel(X_LABEL)
    _set_ylabel_with_formula(ax, r"y = \mathrm{stat}_{d,h,w}(\Delta x)_{dhw}")
    ax.set_title(f"{sequence_id}: aggregate diff statistics")
    _add_notation_caption(fig, y=1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_channel_stats(
    sequence_id: str,
    x: np.ndarray,
    diffs: np.ndarray,
    out_path: Path,
) -> None:
    n_stats = len(STAT_NAMES)
    fig, axes = plt.subplots(n_stats, 1, figsize=(10, 3.4 * n_stats), sharex=True)
    if n_stats == 1:
        axes = [axes]

    n_channels = diffs.shape[1]
    for ax, stat in zip(axes, STAT_NAMES):
        curves = _per_channel_stat_curves(diffs, stat)
        for ch in range(n_channels):
            ax.plot(
                x,
                curves[:, ch],
                color="C0",
                alpha=0.25,
                linewidth=1,
            )
        ax.plot(
            x,
            _aggregate_stat_curve(diffs, stat),
            color="black",
            linewidth=2.5,
            marker="o",
            label=f"${AGGREGATE_FORMULAS[stat]}$",
        )
        _set_ylabel_with_formula(
            ax,
            PER_CHANNEL_FORMULAS[stat],
            short=f"per $d$ ({n_channels} lines)",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(X_LABEL)
    fig.suptitle(
        f"{sequence_id}: per-channel diff statistics\n"
        r"faint: $\mathrm{stat}_{h,w}(\Delta x)_{dhw}$ per $d$; "
        r"black: $\mathrm{stat}_{d,h,w}(\Delta x)_{dhw}$",
        fontsize=10,
    )
    _add_notation_caption(fig, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_l2_and_cosine(
    sequence_id: str,
    x: np.ndarray,
    embeddings: np.ndarray,
    diffs: np.ndarray,
    out_path: Path,
) -> None:
    fig, (ax_l2, ax_cos) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    per_ch_l2 = np.linalg.norm(
        diffs.reshape(diffs.shape[0], diffs.shape[1], -1), axis=2
    )
    for ch in range(per_ch_l2.shape[1]):
        ax_l2.plot(x, per_ch_l2[:, ch], color="C1", alpha=0.2, linewidth=1)
    ax_l2.plot(
        x,
        _l2_norm_curve(diffs),
        marker="o",
        color="C1",
        linewidth=2,
        label=f"${L2_GLOBAL_FORMULA}$",
    )
    _set_ylabel_with_formula(ax_l2, L2_GLOBAL_FORMULA, short="L2")
    ax_l2.set_title(
        f"{sequence_id}: diff L2 norm\n"
        f"faint: ${L2_PER_CHANNEL_FORMULA}$",
        fontsize=10,
    )
    ax_l2.legend(fontsize=8, loc="upper right")
    ax_l2.grid(True, alpha=0.3)

    ax_cos.plot(
        x,
        _cosine_distance_curve(embeddings),
        marker="o",
        color="C2",
        linewidth=2,
        label=f"${COSINE_DISTANCE_FORMULA}$",
    )
    ax_cos.set_xlabel(X_LABEL)
    _set_ylabel_with_formula(ax_cos, COSINE_DISTANCE_FORMULA, short="cosine dist.")
    ax_cos.set_title(r"Embedding cosine distance ($x_i$ vs. $x_{i+1}$)", fontsize=10)
    ax_cos.legend(fontsize=8, loc="upper right")
    ax_cos.grid(True, alpha=0.3)

    _add_notation_caption(fig, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mean_abs_heatmaps(
    sequence_id: str,
    x: np.ndarray,
    diffs: np.ndarray,
    out_path: Path,
) -> None:
    maps = _mean_abs_spatial_maps(diffs)
    n = maps.shape[0]
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    vmax = float(np.max(maps)) if maps.size else 1.0
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < n:
            im = ax.imshow(maps[idx], cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"frame {int(x[idx])}")
            ax.axis("off")
        else:
            ax.axis("off")

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.6,
        label=f"${MEAN_ABS_SPATIAL_FORMULA}$",
    )
    fig.suptitle(
        f"{sequence_id}: mean $|\\Delta x|$ over $d$, per $(h,w)$\n"
        f"${MEAN_ABS_SPATIAL_FORMULA}$",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.01,
        NOTATION_CAPTION,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fraction_above_threshold(
    sequence_id: str,
    x: np.ndarray,
    diffs: np.ndarray,
    out_path: Path,
    thresholds: Tuple[float, ...] = (1e-4, 1e-3, 1e-2),
) -> None:
    flat = np.abs(diffs.reshape(diffs.shape[0], -1))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for thr in thresholds:
        frac = np.mean(flat > thr, axis=1)
        ax.plot(
            x,
            frac,
            marker="o",
            label=rf"$\tau={thr:g}$ in ${FRACTION_ABOVE_FORMULA}$",
        )
    ax.set_xlabel(X_LABEL)
    _set_ylabel_with_formula(ax, FRACTION_ABOVE_FORMULA, short="fraction")
    ax.set_title(
        rf"{sequence_id}: fraction of large $|\Delta x|_{{dhw}}$"
    )
    _add_notation_caption(fig, y=1.02)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_sequences_comparison(
    sequence_stats: Dict[str, Dict[str, np.ndarray]],
    x_by_seq: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False)
    metrics = [
        ("mean", AGGREGATE_FORMULAS["mean"]),
        ("std", AGGREGATE_FORMULAS["std"]),
        ("l2", L2_GLOBAL_FORMULA),
        ("cosine", COSINE_DISTANCE_FORMULA),
    ]

    for ax, (key, formula) in zip(axes.ravel(), metrics):
        for sequence_id, stats in sequence_stats.items():
            ax.plot(
                x_by_seq[sequence_id],
                stats[key],
                marker="o",
                linewidth=2,
                label=sequence_id,
            )
        ax.set_xlabel(X_LABEL)
        _set_ylabel_with_formula(ax, formula)
        ax.set_title(f"${formula}$", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "All sequences: embedding change comparison\n" + NOTATION_CAPTION,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def inspect_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: Path | None = None,
) -> Path:
    """Load dataset, compute diff stats, save plots. Returns output directory."""
    if not fo.dataset_exists(dataset_name):
        raise ValueError(f"Dataset {dataset_name!r} does not exist")

    dataset = fo.load_dataset(dataset_name)
    if EMBEDDING_FIELD not in dataset.get_field_schema():
        raise ValueError(
            f"Dataset {dataset_name!r} has no {EMBEDDING_FIELD!r}; "
            "run make_synthetic_dataset.py first"
        )

    script_dir = Path(__file__).resolve().parent
    out_root = output_dir or _stats_dir(script_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sequence_ids = sorted(dataset.distinct("sequence_id"))
    comparison_stats: Dict[str, Dict[str, np.ndarray]] = {}
    x_by_seq: Dict[str, np.ndarray] = {}

    for sequence_id in sequence_ids:
        seq_dir = out_root / sequence_id
        seq_dir.mkdir(parents=True, exist_ok=True)

        frame_numbers, embeddings = _load_sequence_embeddings(dataset, sequence_id)
        diffs = _consecutive_diffs(embeddings)
        x = _diff_x_axis(frame_numbers)

        logger.info(
            "sequence %s: %d frames, embeddings %s, diffs %s",
            sequence_id,
            len(frame_numbers),
            embeddings.shape,
            diffs.shape,
        )

        comparison_stats[sequence_id] = {
            "mean": _aggregate_stat_curve(diffs, "mean"),
            "std": _aggregate_stat_curve(diffs, "std"),
            "l2": _l2_norm_curve(diffs),
            "cosine": _cosine_distance_curve(embeddings),
        }
        x_by_seq[sequence_id] = x

        plot_aggregate_stats(
            sequence_id, x, diffs, seq_dir / "aggregate_stats.png"
        )
        plot_per_channel_stats(
            sequence_id, x, diffs, seq_dir / "per_channel_stats.png"
        )
        plot_l2_and_cosine(
            sequence_id, x, embeddings, diffs, seq_dir / "l2_and_cosine.png"
        )
        plot_mean_abs_heatmaps(
            sequence_id, x, diffs, seq_dir / "mean_abs_spatial_heatmaps.png"
        )
        plot_fraction_above_threshold(
            sequence_id, x, diffs, seq_dir / "fraction_above_threshold.png"
        )

    plot_all_sequences_comparison(
        comparison_stats, x_by_seq, out_root / "all_sequences_comparison.png"
    )

    logger.info("Saved plots under %s", out_root)
    return out_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SAM2 embedding diff statistics for the synthetic dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"FiftyOne dataset name (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output folder (default: <plugin>/{STATS_DIR_NAME}/)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    out = args.output_dir
    output_dir = Path(out) if out is not None else None
    saved = inspect_dataset(
        dataset_name=args.dataset_name,
        output_dir=output_dir,
    )
    print(f"Plots saved to {saved}")


if __name__ == "__main__":
    main()
