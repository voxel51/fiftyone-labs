import time
from pathlib import Path
import sys

import pytest
import numpy as np

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

_INTENSIVE_DIR = Path(__file__).resolve().parent
PLUGINS_DIR = _INTENSIVE_DIR.parent.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))


TEST_CASES = [
    {"label_indices": "0", "start_frame_number": 1, "end_frame_number": 10},
    {"label_indices": "1", "start_frame_number": 1, "end_frame_number": 10},  # same range
    {"label_indices": "1", "start_frame_number": 9, "end_frame_number": 18},
]


def _assert_propagated_indices(frame_indices_list, tracked):
    """Seed frames keep untracked labels; propagated frames only have tracked indices."""
    if tracked == {0, 1}:
        for frame_indices in frame_indices_list:
            if not frame_indices:
                continue
            idx_set = set(frame_indices)
            assert idx_set <= tracked
            assert idx_set
        return

    seed_positions = {
        pos
        for pos, frame_indices in enumerate(frame_indices_list)
        if frame_indices and (set(frame_indices) - tracked)
    }
    assert len(seed_positions) >= 2

    for pos, frame_indices in enumerate(frame_indices_list):
        if not frame_indices:
            continue
        idx_set = set(frame_indices)
        if pos in seed_positions:
            assert tracked & idx_set
            assert idx_set - tracked
        else:
            assert idx_set <= tracked
            assert idx_set


@pytest.fixture
def image_dataset_view():
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    SELECT_SEQUENCES = ["bike-packing"]
    dataset_view = dataset.match_tags(SELECT_SEQUENCES)
    dataset_view = dataset_view.match(F("frame_number").to_int() < 20)
    return dataset_view


@pytest.fixture
def partially_labeled_image_dataset_view(image_dataset_view):
    if "labels_test_m1" in image_dataset_view._dataset.get_field_schema():
        image_dataset_view._dataset.delete_sample_field(
            "labels_test_m1", error_level=2
        )

    if "labels_test_m1" not in image_dataset_view._dataset.get_field_schema():
        image_dataset_view._dataset.add_sample_field(
            "labels_test_m1",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

    sequences = image_dataset_view.distinct("tags")
    sequences.remove("val")
    new_frame_number = 0
    for seq in sequences:
        seq_slice = image_dataset_view.match_tags(seq).sort_by("frame_number")
        n = len(seq_slice)
        seq_slice.set_values(
            "new_frame_number",
            [new_frame_number + ii for ii in range(n)],
        )
        new_frame_number += n

        for idx in [n // 3, n * 2 // 3]:
            sample = seq_slice.skip(idx).first()
            sample["labels_test_m1"] = sample["ground_truth"]
            sample.save()

    return image_dataset_view


def _run_operator(view, test_case, cache_view):
    ctx = {
        "dataset": view._dataset,
        "view": view,
        "params": {
            "annotation_field": "labels_test_m1",
            "label_indices": test_case["label_indices"],
            "start_frame_number": test_case["start_frame_number"],
            "end_frame_number": test_case["end_frame_number"],
            "sort_field": "new_frame_number",
            "cache_view": cache_view,
        },
    }
    return foo.execute_operator(
        "@51labs/label_propagation/propagate_labels_m1", ctx
    )


def test_cache_population(partially_labeled_image_dataset_view):
    view = partially_labeled_image_dataset_view

    times = []
    for test_case in TEST_CASES:
        t0 = time.perf_counter()
        _run_operator(view, test_case, cache_view=True)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        output_indices = view.values("labels_test_m1.detections.index")
        test_indices = {int(x.strip()) for x in test_case["label_indices"].split(",")}
        _assert_propagated_indices(output_indices, test_indices)
    
    first, second, third = times
    assert second < first*0.75, "second run should have used the cache of the first"
    assert second < third*0.75, "third run is on a different range; currently doesn't use the cache"


def test_no_cache_population(partially_labeled_image_dataset_view):
    view = partially_labeled_image_dataset_view

    times = []
    for test_case in TEST_CASES[:-1]:
        t0 = time.perf_counter()
        _run_operator(view, test_case, cache_view=False)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        output_indices = view.values("labels_test_m1.detections.index")
        test_indices = {int(x.strip()) for x in test_case["label_indices"].split(",")}
        _assert_propagated_indices(output_indices, test_indices)
    
    first, second = times
    assert second > first*0.75, "second run should not see any speedup"
