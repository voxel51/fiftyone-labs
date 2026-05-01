import pytest
import numpy as np
from pathlib import Path
import sys
import os

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

# for voxbox benchmarking, get batch size from env var
BATCH_SIZE = int(os.getenv("BENCHMARK_BATCH_SIZE", 32))
NUM_SEQUENCES = int(os.getenv("BENCHMARK_NUM_SEQUENCES", 5))

_TEST_PKG_DIR = Path(__file__).resolve().parent.parent
PLUGINS_DIR = _TEST_PKG_DIR.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))


@pytest.fixture(params=list(range(NUM_SEQUENCES)))
def image_dataset_view(request):
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/mose-v2",
        split="train",
        # "https://github.com/voxel51/davis-2017",
        # split="validation",
        # format="image",
    )
    sequence_ids = sorted(dataset.distinct("sequence_id"))  # type: ignore[arg-type]
    sequence_id = sequence_ids[request.param]
    view = dataset.match(F("sequence_id") == sequence_id)
    return view


@pytest.fixture
def partially_labeled_image_dataset_view(image_dataset_view):
    for field in ("labels_test", "labels_test_propagated"):
        if field in image_dataset_view._dataset.get_field_schema():
            image_dataset_view._dataset.delete_sample_field(
                field, error_level=2
            )

    image_dataset_view._dataset.add_sample_field(
        "labels_test",
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

        # label only the first frame
        exemplar_sample = seq_slice.first()
        exemplar_sample["labels_test"] = exemplar_sample["ground_truth"]
        exemplar_sample.save()

    return image_dataset_view


def test_propagate_labels_image(partially_labeled_image_dataset_view):
    view = partially_labeled_image_dataset_view
    ctx = {
        "dataset": view._dataset,
        "view": view,
        "params": {
            "input_annotation_field": "labels_test",
            "output_annotation_field": "labels_test_propagated",
            "propagation_method": "sam2",
            "sort_field": "new_frame_number",
            "batch_size": BATCH_SIZE,
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    print(result.result["message"])  # type: ignore[index]


# @pytest.fixture(params=list(range(NUM_SEQUENCES)))
# def video_dataset_view(request):
#     dataset = foz.load_zoo_dataset(
#         "https://github.com/voxel51/davis-2017",
#         dataset_name="davis-2017-video-validation",
#         split="validation",
#         format="video",
#     )
#     sequence_ids = sorted(dataset.distinct("sequence_id"))  # type: ignore[arg-type]
#     sequence_id = sequence_ids[request.param]
#     dataset_view = dataset.match(F("sequence_id") == sequence_id)
#     return dataset_view


# @pytest.fixture
# def partially_labeled_video_dataset_view(video_dataset_view):
#     if (
#         "labels_test"
#         not in video_dataset_view._dataset.get_frame_field_schema()
#     ):
#         video_dataset_view._dataset.add_frame_field(
#             "labels_test",
#             fo.EmbeddedDocumentField,
#             embedded_doc_type=fo.Detections,
#         )
#     for sample in video_dataset_view.iter_samples(autosave=True):
#         for frame_number, frame in sample.frames.items():
#             if frame_number == 1:
#                 frame["labels_test"] = frame["ground_truth"]
#             else:
#                 frame["labels_test"] = fo.Detections(detections=[])

#     return video_dataset_view


# def test_propagate_labels_video(partially_labeled_video_dataset_view):
#     ctx = {
#         "dataset": partially_labeled_video_dataset_view._dataset,
#         "view": partially_labeled_video_dataset_view,
#         "params": {
#             "input_annotation_field": "frames.labels_test",
#             "output_annotation_field": "frames.labels_test_propagated",
#             "propagation_method": "sam2",
#             "sort_field": "frames.frame_number",
#             "batch_size": 1,
#         },
#     }
#     result = foo.execute_operator(
#         "@51labs/label_propagation/propagate_labels", ctx
#     )
#     print(result.result["message"])  # type: ignore[index]
