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

from label_propagation.kp_propagation import propagate_keypoints_cotracker

_SEQUENCE = "blackswan"
_NUM_FRAMES = 20
_SEED_FRAME_IDX = 0
_NUM_SEED_POINTS = 5
_KP_FIELD = "kp_cotracker_seed"
_KP_OUT_FIELD = "kp_cotracker_propagated"


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
    assert len(view) == _NUM_FRAMES, (
        f"Expected {_NUM_FRAMES} frames, got {len(view)}"
    )
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
            points=[[float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))]],
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
