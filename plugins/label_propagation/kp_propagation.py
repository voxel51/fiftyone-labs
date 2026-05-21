import logging
import os
from collections import defaultdict
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


def add_keypoints_field_if_not_exists(dataset: fo.Dataset, field_name: str):
    if str(dataset.media_type) == "video":
        if field_name not in dataset.get_frame_field_schema():  # type: ignore[arg-type]
            dataset.add_frame_field(
                field_name,
                fo.EmbeddedDocumentField,
                embedded_doc_type=fo.Keypoints,
            )
    else:
        if field_name not in dataset.get_field_schema():
            dataset.add_sample_field(
                field_name,
                fo.EmbeddedDocumentField,
                embedded_doc_type=fo.Keypoints,
            )

    singleton = fod.Dataset._instances.get(dataset.name)  # type: ignore[attr-defined]
    if singleton is not None and singleton is not dataset:
        singleton.reload()


def _load_image_rgb(path: str) -> np.ndarray:
    """Load image as (H, W, 3) uint8 RGB array."""
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def propagate_keypoints_cotracker(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
    progress: Optional[bool] = True,
    bidirectional: bool = False,
) -> dict:
    """Propagate keypoints from seed frames to all frames using CoTracker.

    Args:
        view: Dataset or view containing the image sequence.
        input_annotation_field: Field of type fo.Keypoints holding seed labels.
        output_annotation_field: Field to write propagated fo.Keypoints into.
        sort_field: Optional field to sort samples by before propagating.
        progress: Whether to show progress output.
        bidirectional: If True, run CoTracker with backward_tracking=True so
            points are also tracked backwards from their seed frame.
    """
    media_mode = str(view.media_type)
    if media_mode == "group":
        view = view.flatten()
        media_mode = "image"

    if media_mode == "video":
        raise NotImplementedError(
            "Video media type is not yet supported for CoTracker keypoint "
            "propagation; convert to an image sequence first."
        )

    run_view = (
        view.sort_by(sort_field)
        if (sort_field and view.has_field(sort_field))
        else view
    )

    samples = list(run_view.iter_samples(progress=progress))
    if not samples:
        logger.warning("Empty view — nothing to propagate.")
        return {}

    # --- Load all frames into a video tensor ---
    frames: list[np.ndarray] = [
        _load_image_rgb(get_local_path(s)) for s in samples
    ]
    H, W = frames[0].shape[:2]

    video_np = np.stack(frames)  # (T, H, W, 3)
    video_t = (
        torch.from_numpy(video_np).permute(0, 3, 1, 2).float().unsqueeze(0)
    )  # (1, T, 3, H, W)

    # --- Collect seed queries from labeled frames ---
    # Key: (label, index) — keep earliest occurrence so each tracked "object"
    # has exactly one query entry in CoTracker.
    seed_meta: dict[tuple, tuple] = {}  # (label, index) -> (frame_idx, points)
    for frame_idx, sample in enumerate(samples):
        kps: Optional[fol.Keypoints] = sample.get_field(input_annotation_field)
        if kps is None or not kps.keypoints:
            continue
        for kp in kps.keypoints:
            key = (kp.label, kp.index)
            if key not in seed_meta:
                seed_meta[key] = (frame_idx, list(kp.points or []))

    if not seed_meta:
        logger.warning(
            "No keypoints found in '%s'. Nothing to propagate.",
            input_annotation_field,
        )
        return {}

    # Build flat queries tensor and remember which slice belongs to each key.
    queries_list: list[list[float]] = []
    kp_slices: dict[tuple, tuple[int, int]] = {}  # (label, index) -> (start, end)
    for key, (frame_idx, points) in seed_meta.items():
        start = len(queries_list)
        for x_norm, y_norm in points:
            queries_list.append([float(frame_idx), x_norm * W, y_norm * H])
        kp_slices[key] = (start, len(queries_list))

    queries_t = torch.tensor(
        queries_list, dtype=torch.float32
    ).unsqueeze(0)  # (1, N, 3)

    # --- Run CoTracker ---
    model = load_cotracker()
    device = _cotracker_device(model)
    video_t = video_t.to(device)
    queries_t = queries_t.to(device)

    with torch.no_grad():
        tracks, visibilities = model(
            video_t, queries=queries_t, backward_tracking=bidirectional
        )

    # tracks: (1, T, N, 2) pixel coords (x, y)
    # visibilities: (1, T, N) bool or float
    tracks_np = tracks[0].cpu().numpy()  # (T, N, 2)
    vis_np = visibilities[0].cpu().numpy()  # (T, N)

    # --- Register output field and write results ---
    add_keypoints_field_if_not_exists(view._dataset, output_annotation_field)

    for t_idx, sample in enumerate(samples):
        kp_list: list[fol.Keypoint] = []
        for (label, kp_index), (start, end) in kp_slices.items():
            points_out: list[list[float]] = []
            confidences_out: list[float] = []
            for q in range(start, end):
                x_px, y_px = float(tracks_np[t_idx, q, 0]), float(
                    tracks_np[t_idx, q, 1]
                )
                vis = float(vis_np[t_idx, q])
                points_out.append(
                    [
                        float(np.clip(x_px / W, 0.0, 1.0)),
                        float(np.clip(y_px / H, 0.0, 1.0)),
                    ]
                )
                confidences_out.append(vis)
            kp_list.append(
                fol.Keypoint(
                    label=label,
                    index=kp_index,
                    points=points_out,
                    confidence=confidences_out,
                )
            )
        sample.set_field(
            output_annotation_field, fol.Keypoints(keypoints=kp_list)
        )
        sample.save()

    return {}
