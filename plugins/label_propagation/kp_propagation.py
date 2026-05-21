import logging
from typing import Union, Optional, Any

import numpy as np
import torch

import fiftyone as fo
import fiftyone.core.dataset as fod
import fiftyone.core.labels as fol

from .utils import get_local_path

logger = logging.getLogger(__name__)

_COTRACKER_CACHE: dict[str, Any] = {}


def load_cotracker() -> Any:
    if "model" in _COTRACKER_CACHE:
        return _COTRACKER_CACHE["model"]
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model.eval()
    _COTRACKER_CACHE["model"] = model
    return model


def _cotracker_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _load_image_rgb(path: str) -> np.ndarray:
    """Load image as (H, W, 3) uint8 RGB array."""
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_video_np(samples: list) -> tuple[np.ndarray, int, int]:
    """Stack samples into (T, H, W, 3) uint8 numpy. Returns (video, H, W)."""
    frames = [_load_image_rgb(get_local_path(s)) for s in samples]
    H, W = frames[0].shape[:2]
    return np.stack(frames), H, W


def _run_cotracker_tracks(
    video_np: np.ndarray,
    queries_list: list,
    bidirectional: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run CoTracker offline on video_np with the given point queries.

    Args:
        video_np: (T, H, W, 3) uint8.
        queries_list: list of [frame_idx, x_px, y_px].
        bidirectional: pass backward_tracking=True to CoTracker.

    Returns:
        tracks: (T, N, 2) float32 pixel coords.
        vis:    (T, N)    float32 visibility scores.
    """
    model = load_cotracker()
    device = _cotracker_device(model)
    video_t = (
        torch.from_numpy(video_np)
        .permute(0, 3, 1, 2)
        .float()
        .unsqueeze(0)
        .to(device)
    )  # (1, T, 3, H, W)
    queries_t = (
        torch.tensor(queries_list, dtype=torch.float32).unsqueeze(0).to(device)
    )  # (1, N, 3)
    with torch.no_grad():
        tracks, vis = model(
            video_t, queries=queries_t, backward_tracking=bidirectional
        )
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy()


def _add_label_field_if_not_exists(
    dataset: fo.Dataset, field_name: str, embedded_doc_type
):
    if str(dataset.media_type) == "video":
        if field_name not in dataset.get_frame_field_schema():  # type: ignore[arg-type]
            dataset.add_frame_field(
                field_name,
                fo.EmbeddedDocumentField,
                embedded_doc_type=embedded_doc_type,
            )
    else:
        if field_name not in dataset.get_field_schema():
            dataset.add_sample_field(
                field_name,
                fo.EmbeddedDocumentField,
                embedded_doc_type=embedded_doc_type,
            )
    singleton = fod.Dataset._instances.get(dataset.name)  # type: ignore[attr-defined]
    if singleton is not None and singleton is not dataset:
        singleton.reload()


def add_keypoints_field_if_not_exists(dataset: fo.Dataset, field_name: str):
    _add_label_field_if_not_exists(dataset, field_name, fo.Keypoints)


def add_polylines_field_if_not_exists(dataset: fo.Dataset, field_name: str):
    _add_label_field_if_not_exists(dataset, field_name, fo.Polylines)


def _sorted_run_view(view, sort_field):
    return (
        view.sort_by(sort_field)
        if (sort_field and view.has_field(sort_field))
        else view
    )


def propagate_keypoints_cotracker(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
    progress: Optional[bool] = True,
    bidirectional: bool = False,
) -> dict:
    """Propagate fo.Keypoints from seed frames to all frames using CoTracker.

    Each (label, index) pair in the seed field becomes a set of independent
    CoTracker queries — one per point. Confidence is populated from CoTracker
    visibility scores.
    """
    media_mode = str(view.media_type)
    if media_mode == "group":
        view = view.flatten()
        media_mode = "image"
    if media_mode == "video":
        raise NotImplementedError(
            "Video media type not yet supported; convert to an image sequence."
        )

    samples = list(
        _sorted_run_view(view, sort_field).iter_samples(progress=progress)
    )
    if not samples:
        logger.warning("Empty view — nothing to propagate.")
        return {}

    video_np, H, W = _load_video_np(samples)

    # (label, index) -> (frame_idx, [(x_norm, y_norm), ...])  — earliest occurrence
    seed_meta: dict[tuple, tuple] = {}
    for frame_idx, sample in enumerate(samples):
        kps: Optional[fol.Keypoints] = sample.get_field(input_annotation_field)
        if kps is None or not kps.keypoints:
            continue
        for kp in kps.keypoints:
            key = (kp.label, kp.index)
            if key not in seed_meta:
                seed_meta[key] = (frame_idx, list(kp.points or []))

    if not seed_meta:
        logger.warning("No keypoints found in '%s'.", input_annotation_field)
        return {}

    queries_list: list[list[float]] = []
    kp_slices: dict[tuple, tuple[int, int]] = {}
    for key, (frame_idx, points) in seed_meta.items():
        start = len(queries_list)
        for x_n, y_n in points:
            queries_list.append(
                [float(frame_idx), float(x_n) * W, float(y_n) * H]
            )
        kp_slices[key] = (start, len(queries_list))

    tracks_np, vis_np = _run_cotracker_tracks(
        video_np, queries_list, bidirectional
    )

    add_keypoints_field_if_not_exists(view._dataset, output_annotation_field)

    for t_idx, sample in enumerate(samples):
        kp_list: list[fol.Keypoint] = []
        for (label, kp_index), (start, end) in kp_slices.items():
            pts_out, conf_out = [], []
            for q in range(start, end):
                x_px = float(tracks_np[t_idx, q, 0])
                y_px = float(tracks_np[t_idx, q, 1])
                pts_out.append(
                    [
                        float(np.clip(x_px / W, 0.0, 1.0)),
                        float(np.clip(y_px / H, 0.0, 1.0)),
                    ]
                )
                conf_out.append(float(vis_np[t_idx, q]))
            kp_list.append(
                fol.Keypoint(
                    label=label,
                    index=kp_index,
                    points=pts_out,
                    confidence=conf_out,
                )
            )
        sample.set_field(
            output_annotation_field, fol.Keypoints(keypoints=kp_list)
        )
        sample.save()

    return {}


def propagate_polylines_cotracker(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
    progress: Optional[bool] = True,
    bidirectional: bool = False,
) -> dict:
    """Propagate fo.Polylines (and polygons) from seed frames using CoTracker.

    Each vertex of each polyline path becomes an independent CoTracker query.
    After tracking, the points are rejoined in the original order, preserving
    path structure, label, index, closed, and filled attributes.
    """
    media_mode = str(view.media_type)
    if media_mode == "group":
        view = view.flatten()
        media_mode = "image"
    if media_mode == "video":
        raise NotImplementedError(
            "Video media type not yet supported; convert to an image sequence."
        )

    samples = list(
        _sorted_run_view(view, sort_field).iter_samples(progress=progress)
    )
    if not samples:
        logger.warning("Empty view — nothing to propagate.")
        return {}

    video_np, H, W = _load_video_np(samples)

    # (label, index) -> {path_shapes, query_start, query_end, closed, filled}
    # path_shapes: number of points per path (to reconstruct nested structure)
    seed_meta: dict[tuple, dict] = {}
    queries_list: list[list[float]] = []

    for frame_idx, sample in enumerate(samples):
        plines: Optional[fol.Polylines] = sample.get_field(
            input_annotation_field
        )
        if plines is None or not plines.polylines:
            continue
        for pl in plines.polylines:
            key = (pl.label, pl.index)
            if key in seed_meta:
                continue
            paths = pl.points or []
            start = len(queries_list)
            path_shapes = []
            for path in paths:
                path_shapes.append(len(path))
                for x_n, y_n in path:
                    queries_list.append(
                        [float(frame_idx), float(x_n) * W, float(y_n) * H]
                    )
            seed_meta[key] = {
                "path_shapes": path_shapes,
                "query_start": start,
                "query_end": len(queries_list),
                "closed": bool(getattr(pl, "closed", False)),
                "filled": bool(getattr(pl, "filled", False)),
            }

    if not queries_list:
        logger.warning("No polylines found in '%s'.", input_annotation_field)
        return {}

    tracks_np, _ = _run_cotracker_tracks(video_np, queries_list, bidirectional)

    add_polylines_field_if_not_exists(view._dataset, output_annotation_field)

    for t_idx, sample in enumerate(samples):
        pl_list: list[fol.Polyline] = []
        for (label, kp_index), meta in seed_meta.items():
            q = meta["query_start"]
            paths_out = []
            for n_pts in meta["path_shapes"]:
                path_out = []
                for _ in range(n_pts):
                    x_px = float(tracks_np[t_idx, q, 0])
                    y_px = float(tracks_np[t_idx, q, 1])
                    path_out.append(
                        [
                            float(np.clip(x_px / W, 0.0, 1.0)),
                            float(np.clip(y_px / H, 0.0, 1.0)),
                        ]
                    )
                    q += 1
                paths_out.append(path_out)
            pl_list.append(
                fol.Polyline(
                    label=label,
                    index=kp_index,
                    points=paths_out,
                    closed=meta["closed"],
                    filled=meta["filled"],
                )
            )
        sample.set_field(
            output_annotation_field, fol.Polylines(polylines=pl_list)
        )
        sample.save()

    return {}
