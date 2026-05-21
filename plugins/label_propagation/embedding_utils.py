import logging
import os
from typing import Dict, List, Optional

import numpy as np

import fiftyone as fo
import fiftyone.core.dataset as fod
import fiftyone.core.media as focm
from PIL import Image

from .propagation import load_local_sam2
from .utils import get_local_path


logger = logging.getLogger(__name__)


class _ImageSamplesAsVideoFrames:
    """Presents a list of image samples as a pseudo-video for SAM2 init_state.

    Must match the contract expected by ``load_fiftyone_video_frames`` in
    ``sam2_local`` (tuple ``(sample, reader)`` passed where a video path would
    normally go), because the SAM2 predictor keeps a reference to the patched
    ``load_video_frames`` after model load.
    """

    media_type = focm.IMAGE

    def __init__(self, frames: List[fo.Sample]):
        self._frames = list(frames)

    @property
    def frames(self):
        return {ii + 1: ff for ii, ff in enumerate(self._frames)}


def extract_spatial_embeddings(
    sam2_predictor,
    inference_state,
    frame_idx: int,
    feature_level: int = 0,
):
    """
    Extract patch-wise backbone embeddings for one frame in ``inference_state``.

    Args:
        sam2_predictor: SAM2 video predictor (``SegmentAnything2VideoModel.model``).
        inference_state: State from ``init_state``.
        frame_idx: 0-based index in the frame order used for ``init_state``.
        feature_level: SAM2 feature pyramid level (0 = highest resolution).

    Returns:
        A numpy array with shape (C, H, W) for the given feature level.
        C = number of channels, H = embedding height, W = embedding width
    """
    import torch

    logger.debug("Extracting patch embeddings for frame_idx %s...", frame_idx)

    with torch.no_grad():
        _, _, vision_feats, _, feat_sizes = sam2_predictor._get_image_feature(
            inference_state,
            frame_idx,
            batch_size=1,
        )
        vision_feat = vision_feats[feature_level]
        feat_size = feat_sizes[feature_level]

        # feat shape: (HW, B, C) -> reshape to (B, C, H, W)
        H, W = feat_size
        B, C = vision_feat.shape[1], vision_feat.shape[2]
        spatial_feat = vision_feat.permute(1, 2, 0).view(B, C, H, W)
        spatial_feat = spatial_feat.squeeze(0).cpu().numpy()

    logger.debug("Extracted spatial embeddings for frame_idx %s", frame_idx)
    return spatial_feat


def get_sam2_embeddings(view: fo.DatasetView):
    assert str(view.media_type) == "image", "Only image mode is supported"

    fo_model = load_local_sam2(media_mode="image")
    sam2_predictor = fo_model.model

    ordered = list(view.sort_by("frame_number").iter_samples())
    filepath_to_idx: Dict[str, int] = {
        os.path.abspath(get_local_path(s)): i for i, s in enumerate(ordered)
    }

    mock_sample = _ImageSamplesAsVideoFrames(ordered)
    hh, ww = 0, 0
    if len(ordered) > 0:
        sample_filepath = os.path.abspath(get_local_path(ordered[0]))
        hh, ww = Image.open(sample_filepath).size
    mock_reader = type("_Reader", (), {"frame_size": (ww, hh)})()

    inference_state = sam2_predictor.init_state((mock_sample, mock_reader))

    field_name = "sam2_backbone_embeddings"
    dataset = view._dataset
    if field_name not in dataset.get_field_schema():
        dataset.add_sample_field(field_name, fo.ArrayField)
        singleton = fod.Dataset._instances.get(dataset.name)  # type: ignore[attr-defined]
        if singleton is not None and singleton is not dataset:
            singleton.reload()

    def set_sample_embedding(sample: fo.Sample):
        key = os.path.abspath(get_local_path(sample))
        frame_idx = filepath_to_idx[key]
        arr = extract_spatial_embeddings(
            sam2_predictor, inference_state, frame_idx
        )
        sample[field_name] = arr

    for _ in view.map_samples(
        set_sample_embedding,
        save=True,
        num_workers=1,
    ):
        pass

    return view


def default_patch_neighborhood_radius(height: int, width: int) -> int:
    """Spatial search radius in the source frame (matches exemplar-frames plugin)."""
    return max(height, width) // 10


def hausdorff_distance_map(
    emb_target: np.ndarray,
    emb_source: np.ndarray,
    patch_nbd: Optional[int] = None,
    search_entire_source: bool = False,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    """
    Asymmetric Hausdorff distance map from ``emb_source`` to ``emb_target``.

    For each spatial location ``(h, w)`` in ``emb_target`` (frame ``i+1``), take the
    ``D``-vector patch ``emb_target[:, h, w]`` and assign the minimum L2 distance to
    any patch in ``emb_source`` (frame ``i``). By default the search is restricted to
    a square neighborhood around ``(h, w)`` in the source map (as in the video
    exemplar frames plugin); set ``search_entire_source=True`` to search globally.

    Args:
        emb_target: Later-frame embedding ``(D, H, W)``.
        emb_source: Earlier-frame embedding ``(D, H, W)``.
        patch_nbd: Half-width of the source neighborhood (pixels). Defaults to
            ``max(H, W) // 10``.
        search_entire_source: If True, search all of ``emb_source`` per target patch.
        show_progress: If True, show a tqdm bar over spatial locations.
        progress_desc: Optional tqdm description.

    Returns:
        Array of shape ``(H, W)`` with per-patch Hausdorff distances.
    """
    if emb_target.shape != emb_source.shape:
        raise ValueError(
            f"Shape mismatch: target {emb_target.shape} vs source {emb_source.shape}"
        )
    _, height, width = emb_target.shape
    if patch_nbd is None:
        patch_nbd = default_patch_neighborhood_radius(height, width)

    diff_map = np.zeros((height, width), dtype=np.float32)
    positions = [(hh, ww) for hh in range(height) for ww in range(width)]
    if show_progress:
        from tqdm import tqdm

        positions = tqdm(
            positions,
            desc=progress_desc or "Hausdorff map (per location)",
            leave=False,
        )

    for hh, ww in positions:
        patch_target = emb_target[:, hh, ww]
        if search_entire_source:
            neighborhood = emb_source
        else:
            neighborhood = emb_source[
                :,
                max(0, hh - patch_nbd) : min(height, hh + patch_nbd),
                max(0, ww - patch_nbd) : min(width, ww + patch_nbd),
            ]
        dists = np.linalg.norm(
            neighborhood - patch_target[:, np.newaxis, np.newaxis],
            axis=0,
        )
        diff_map[hh, ww] = float(np.min(dists))
    return diff_map


def asymmetric_hausdorff_max_distance(
    emb_target: np.ndarray,
    emb_source: np.ndarray,
    patch_nbd: Optional[int] = None,
    search_entire_source: bool = False,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> float:
    """
    Max over the asymmetric Hausdorff distance map.

    Args:
        emb_target: Later-frame embedding ``(D, H, W)``.
        emb_source: Earlier-frame embedding ``(D, H, W)``.
        patch_nbd: Neighborhood radius passed to :func:`hausdorff_distance_map`.
        search_entire_source: If True, search all source locations per target patch.
        show_progress: Forwarded to :func:`hausdorff_distance_map`.
        progress_desc: Forwarded to :func:`hausdorff_distance_map`.

    Returns:
        ``max_{h,w} min_{h',w'} ||emb_target[:,h,w] - emb_source[:,h',w']||_2``
        (with the inner min over the chosen search region).
    """
    return float(
        np.max(
            hausdorff_distance_map(
                emb_target,
                emb_source,
                patch_nbd=patch_nbd,
                search_entire_source=search_entire_source,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )
        )
    )


def consecutive_hausdorff_distance_maps(
    embeddings: np.ndarray,
    patch_nbd: Optional[int] = None,
    search_entire_source: bool = False,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    """
    Hausdorff distance maps for each consecutive embedding pair.

    Args:
        embeddings: Stack of shape ``(N, D, H, W)``.
        patch_nbd: Neighborhood radius for :func:`hausdorff_distance_map`.
        search_entire_source: If True, search all source locations per target patch.
        show_progress: If True, tqdm over frame transitions (and per-map locations).
        progress_desc: Optional prefix for tqdm descriptions.

    Returns:
        Array of shape ``(N - 1, H, W)``.
    """
    if embeddings.ndim != 4 or embeddings.shape[0] < 2:
        raise ValueError(f"Expected (N>=2, D, H, W), got {embeddings.shape}")

    n_pairs = embeddings.shape[0] - 1
    pair_indices = range(n_pairs)
    if show_progress:
        from tqdm import tqdm

        pair_indices = tqdm(
            pair_indices,
            desc=progress_desc or "Hausdorff maps (frame pairs)",
        )

    maps = []
    for idx in pair_indices:
        desc = None
        if show_progress and progress_desc:
            desc = f"{progress_desc}: pair {idx}->{idx + 1}"
        maps.append(
            hausdorff_distance_map(
                embeddings[idx + 1],
                embeddings[idx],
                patch_nbd=patch_nbd,
                search_entire_source=search_entire_source,
                show_progress=False,
                progress_desc=desc,
            )
        )
    return np.stack(maps, axis=0)


def consecutive_asymmetric_hausdorff_max(
    embeddings: np.ndarray,
    patch_nbd: Optional[int] = None,
    search_entire_source: bool = False,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    """
    Per-transition asymmetric Hausdorff max (length ``N - 1``).

    Args:
        embeddings: Stack of shape ``(N, D, H, W)``.
        patch_nbd: Neighborhood radius for :func:`hausdorff_distance_map`.
        search_entire_source: If True, search all source locations per target patch.
        show_progress: Forwarded to :func:`consecutive_hausdorff_distance_maps`.
        progress_desc: Forwarded to :func:`consecutive_hausdorff_distance_maps`.

    Returns:
        1D array of length ``N - 1``.
    """
    maps = consecutive_hausdorff_distance_maps(
        embeddings,
        patch_nbd=patch_nbd,
        search_entire_source=search_entire_source,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    return np.max(maps, axis=(1, 2))
