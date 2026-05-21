"""CoTracker keypoint propagation test on a DAVIS validation sequence.

Seeds 5 random points on the first frame (each with a distinct index) and
verifies all are propagated to every frame in the sequence. No ground-truth
comparison — we just check that the output is well-formed and coordinates
stay in [0, 1].
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.core.expressions import ViewField as F

sys.path.insert(0, str(Path(__file__).parents[3] / "plugins"))

import cv2

from label_propagation.kp_propagation import (
    propagate_keypoints_cotracker,
    propagate_polylines_cotracker,
)

_SEQUENCE = "blackswan"
_NUM_FRAMES = 20
_SEED_FRAME_IDX = 0
_NUM_SEED_POINTS = 5
_KP_FIELD = "kp_cotracker_seed"
_KP_OUT_FIELD = "kp_cotracker_propagated"
_POLY_FIELD = "poly_cotracker_seed"
_POLY_OUT_FIELD = "poly_cotracker_propagated"
_NUM_POLYLINES = 2
_POLY_PTS = 4  # hull vertices per polyline


@pytest.fixture(scope="module")
def davis_image_view():
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    view = (
        dataset.match_tags(_SEQUENCE)
        .sort_by("frame_number")
        .limit(_NUM_FRAMES)
    )
    assert (
        len(view) == _NUM_FRAMES
    ), f"Expected {_NUM_FRAMES} frames, got {len(view)}"
    return view


@pytest.fixture(scope="module")
def seeded_view(davis_image_view):
    """Attach 5 random Keypoints (each with a unique index) to the first frame."""
    dataset = davis_image_view._dataset

    for field in (_KP_FIELD, _KP_OUT_FIELD):
        if field in dataset.get_field_schema():
            dataset.delete_sample_field(field, error_level=2)

    dataset.add_sample_field(
        _KP_FIELD,
        fo.EmbeddedDocumentField,
        embedded_doc_type=fo.Keypoints,
    )

    rng = np.random.default_rng(42)
    keypoints = [
        fo.Keypoint(
            label="point",
            index=i,
            points=[
                [float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))]
            ],
        )
        for i in range(_NUM_SEED_POINTS)
    ]

    seed_sample = davis_image_view.skip(_SEED_FRAME_IDX).first()
    seed_sample[_KP_FIELD] = fo.Keypoints(keypoints=keypoints)
    seed_sample.save()

    return davis_image_view


def test_cotracker_propagates_to_all_frames(seeded_view):
    propagate_keypoints_cotracker(
        view=seeded_view,
        input_annotation_field=_KP_FIELD,
        output_annotation_field=_KP_OUT_FIELD,
        sort_field="frame_number",
        progress=True,
        bidirectional=False,
    )

    out_kps = seeded_view.values(f"{_KP_OUT_FIELD}.keypoints")

    # Every frame should have exactly _NUM_SEED_POINTS Keypoint objects
    assert all(
        kps is not None and len(kps) == _NUM_SEED_POINTS for kps in out_kps
    ), f"Expected {_NUM_SEED_POINTS} keypoints per frame"

    # Each Keypoint has exactly one point with coordinates in [0, 1]
    for frame_kps in out_kps:
        for kp in frame_kps:
            assert len(kp.points) == 1, "Expected one point per Keypoint"
            x, y = kp.points[0]
            assert 0.0 <= x <= 1.0, f"x={x} out of [0, 1]"
            assert 0.0 <= y <= 1.0, f"y={y} out of [0, 1]"

    # session = fo.launch_app(seeded_view)
    # breakpoint()
    # session.close()


def test_cotracker_labels_and_indices_preserved(seeded_view):
    out_kps = seeded_view.values(f"{_KP_OUT_FIELD}.keypoints")
    expected_indices = set(range(_NUM_SEED_POINTS))
    for frame_kps in out_kps:
        assert {kp.label for kp in frame_kps} == {"point"}
        assert {kp.index for kp in frame_kps} == expected_indices


def test_cotracker_confidence_populated(seeded_view):
    out_kps = seeded_view.values(f"{_KP_OUT_FIELD}.keypoints")
    for frame_kps in out_kps:
        for kp in frame_kps:
            assert kp.confidence is not None and len(kp.confidence) == 1
            assert 0.0 <= kp.confidence[0] <= 1.0


# ---------------------------------------------------------------------------
# Polyline helpers
# ---------------------------------------------------------------------------


def _sample_boundary_pts(det, img_h, img_w, n_pts=4, hull_start_frac=0.0):
    """Sample n_pts convex hull vertices from det's mask boundary.

    hull_start_frac offsets the starting position around the hull [0, 1) so
    multiple calls on the same detection yield distinct point sets.
    Returns list of [x_norm, y_norm] in image coords, or None.
    """
    mask = det.mask
    if mask is None:
        return None
    mask_h, mask_w = mask.shape[:2]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))[
        :, 0, :
    ]  # (N, 2)
    n_hull = len(hull)
    start = int(hull_start_frac * n_hull) % n_hull
    # Evenly spaced indices around the hull, starting at `start`
    indices = sorted(
        {(start + round(j * n_hull / n_pts)) % n_hull for j in range(n_pts)}
    )
    x0_n, y0_n, w_n, h_n = det.bounding_box
    return [
        [
            float(np.clip(x0_n + (xm / mask_w) * w_n, 0.0, 1.0)),
            float(np.clip(y0_n + (ym / mask_h) * h_n, 0.0, 1.0)),
        ]
        for xm, ym in hull[indices]
    ]


def _seed_polylines_on_view(view, field_name, n_pts=4, seed=0):
    """Seed 2 Polylines on the first frame.

    Instance 0 (index=0): n_pts convex hull vertices from the first
    ground_truth detection's mask boundary.
    Instance 1 (index=1): a random triangle placed anywhere in the image,
    unrelated to any mask.
    """
    rng = np.random.default_rng(seed)
    dataset = view._dataset
    if field_name in dataset.get_field_schema():
        dataset.delete_sample_field(field_name, error_level=2)
    dataset.add_sample_field(
        field_name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Polylines
    )

    first = view.sort_by("frame_number").first()
    if first.metadata and first.metadata.height:
        img_h, img_w = first.metadata.height, first.metadata.width
    else:
        img = cv2.imread(first.filepath)
        img_h, img_w = img.shape[:2]

    detections = (
        (first.ground_truth.detections or []) if first.ground_truth else []
    )
    assert detections, "No ground_truth detections on the first frame."

    hull_pts = _sample_boundary_pts(
        detections[0], img_h, img_w, n_pts=n_pts, hull_start_frac=0.0
    )
    assert hull_pts is not None, "Could not sample hull points."

    triangle_pts = [
        [float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95))]
        for _ in range(3)
    ]

    first[field_name] = fo.Polylines(polylines=[
        fo.Polyline(label=detections[0].label, index=0, points=[hull_pts], closed=True, filled=False),
        fo.Polyline(label="random_triangle", index=1, points=[triangle_pts], closed=True, filled=False),
    ])
    first.save()


# ---------------------------------------------------------------------------
# Polyline fixture + test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeded_polylines_view(davis_image_view):
    for field in (_POLY_FIELD, _POLY_OUT_FIELD):
        if field in davis_image_view._dataset.get_field_schema():
            davis_image_view._dataset.delete_sample_field(field, error_level=2)

    _seed_polylines_on_view(davis_image_view, _POLY_FIELD, n_pts=_POLY_PTS)
    return davis_image_view


def test_cotracker_propagates_polylines(seeded_polylines_view):
    propagate_polylines_cotracker(
        view=seeded_polylines_view,
        input_annotation_field=_POLY_FIELD,
        output_annotation_field=_POLY_OUT_FIELD,
        sort_field="frame_number",
        progress=True,
        bidirectional=False,
    )

    # Verify the last frame has both polyline instances with the right indices
    last_sample = seeded_polylines_view.sort_by("frame_number").last()
    out = last_sample[_POLY_OUT_FIELD]
    assert out is not None and len(out.polylines) == _NUM_POLYLINES

    assert {pl.index for pl in out.polylines} == set(range(_NUM_POLYLINES))

    for pl in out.polylines:
        assert len(pl.points) == 1, "Expected one path per polyline"
        for x, y in pl.points[0]:
            assert 0.0 <= x <= 1.0, f"x={x} out of [0, 1]"
            assert 0.0 <= y <= 1.0, f"y={y} out of [0, 1]"
    
    # session = fo.launch_app(seeded_polylines_view)
    # breakpoint()
    # session.close()
    
