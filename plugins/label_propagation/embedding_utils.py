import logging
import os
from typing import Dict, List

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
