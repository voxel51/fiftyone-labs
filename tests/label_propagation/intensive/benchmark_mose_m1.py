import pytest
from pathlib import Path
import sys
import os

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

NUM_SEQUENCES = int(os.getenv("BENCHMARK_NUM_SEQUENCES", 5))
NUM_FRAMES = int(os.getenv("BENCHMARK_NUM_FRAMES", 0))
PROPAGATION_METHOD = os.getenv("BENCHMARK_PROPAGATION_METHOD", "sam2_tiny")

_TEST_PKG_DIR = Path(__file__).resolve().parent.parent
PLUGINS_DIR = _TEST_PKG_DIR.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))


@pytest.fixture(params=list(range(NUM_SEQUENCES)))
def image_dataset_view(request):
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/mose-v2",
        split="train",
    )
    sequence_ids = sorted(dataset.distinct("sequence_id"))  # type: ignore[arg-type]
    sequence_id = sequence_ids[request.param]
    view = dataset.match(F("sequence_id") == sequence_id)
    return view


@pytest.fixture
def partially_labeled_image_dataset_view(image_dataset_view):
    if "labels_test_m1" in image_dataset_view._dataset.get_field_schema():
        image_dataset_view._dataset.delete_sample_field(
            "labels_test_m1", error_level=2
        )

    image_dataset_view._dataset.add_sample_field(
        "labels_test_m1",
        fo.EmbeddedDocumentField,
        embedded_doc_type=fo.Detections,
    )

    sequences = image_dataset_view.distinct("sequence_id")
    new_frame_number = 0
    for seq in sequences:
        seq_slice = image_dataset_view.match(F("sequence_id") == seq).sort_by(
            "frame_number"
        )
        n = len(seq_slice)
        seq_slice.set_values(
            "new_frame_number",
            [new_frame_number + ii for ii in range(n)],
        )
        new_frame_number += n

        exemplar_sample = seq_slice.first()
        exemplar_sample["labels_test_m1"] = exemplar_sample["ground_truth"]
        exemplar_sample.save()

    return image_dataset_view


@pytest.fixture(params=list(range(NUM_SEQUENCES)))
def video_dataset_view(request):
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        dataset_name="davis-2017-video-validation",
        split="validation",
        format="video",
    )
    sequence_ids = sorted(dataset.distinct("sequence_id"))  # type: ignore[arg-type]
    sequence_id = sequence_ids[request.param]
    dataset_view = dataset.match(F("sequence_id") == sequence_id)
    if NUM_FRAMES > 0:
        dataset_view = dataset_view.match_frames(F("frame_number") <= NUM_FRAMES)
    return dataset_view


@pytest.fixture
def partially_labeled_video_dataset_view(video_dataset_view):
    if (
        "labels_test_m1"
        not in video_dataset_view._dataset.get_frame_field_schema()
    ):
        video_dataset_view._dataset.add_frame_field(
            "labels_test_m1",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )
    (
        video_dataset_view.set_field(
            "frames.labels_test_m1", fo.Detections(detections=[])
        ).save()
    )
    for sample in video_dataset_view:
        frame_numbers = sorted(sample.frames.keys())
        n = len(frame_numbers)
        for idx in [n // 3, n * 2 // 3]:
            fn = frame_numbers[idx]
            frame = sample.frames[fn]
            frame["labels_test_m1"] = frame["ground_truth"]
        sample.save()

    return video_dataset_view


def test_propagate_labels_image(partially_labeled_image_dataset_view):
    if PROPAGATION_METHOD == "sam3":
        pytest.skip("SAM3 M1 propagation is video-only")

    view = partially_labeled_image_dataset_view
    n = len(view.sort_by("new_frame_number"))

    ctx = {
        "dataset": view._dataset,
        "view": view,
        "params": {
            "annotation_field": "labels_test_m1",
            "label_indices": "0",
            "start_frame_number": 1,
            "end_frame_number": n,
            "sort_field": "new_frame_number",
            "propagation_method": PROPAGATION_METHOD,
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels_m1", ctx
    )
    print(result.result["message"])  # type: ignore[index]


def test_propagate_labels_video(partially_labeled_video_dataset_view):
    n = len(partially_labeled_video_dataset_view.first().frames)

    ctx = {
        "dataset": partially_labeled_video_dataset_view._dataset,
        "view": partially_labeled_video_dataset_view,
        "params": {
            "annotation_field": "frames.labels_test_m1",
            "label_indices": "0",
            "start_frame_number": 1,
            "end_frame_number": n,
            "propagation_method": PROPAGATION_METHOD,
        },
    }
    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels_m1", ctx
    )
    print(result.result["message"])  # type: ignore[index]
