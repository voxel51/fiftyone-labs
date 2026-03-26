"""Unit tests for functions in the label_propagation plugin."""

from __future__ import annotations

import pytest
from pathlib import Path
import sys
import numpy as np
import cv2

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz
from fiftyone.core.expressions import ViewField as F

_TEST_PKG_DIR = Path(__file__).resolve().parent
PLUGINS_DIR = _TEST_PKG_DIR.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from label_propagation.exemplars import frame_discontinuity  # type: ignore
from label_propagation.utils import get_local_path  # type: ignore


@pytest.fixture(params=["dance-twirl", "motocross-jump", "scooter-black"])
def image_dataset_view(request):
    sequence = request.param
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    dataset_view = dataset.match_tags([sequence]).sort_by("frame_number")

    # if "labels_test" in dataset_view._dataset.get_field_schema():
    #     dataset_view._dataset.delete_sample_field(
    #         "labels_test", error_level=2
    #     )
    # dataset_view._dataset.add_sample_field(
    #     "labels_test",
    #     fo.EmbeddedDocumentField,
    #     embedded_doc_type=fo.Detections,
    # )

    # for ii, sample in enumerate(dataset_view.iter_samples()):
    #     if ii%2 == 0:
    #         sample["labels_test"] = sample["ground_truth"]
    #         sample.save()

    # set all labels_test to concrete ground_truth label values
    dataset_view.set_values("labels_test", dataset_view.values("ground_truth"))
    dataset_view.save()

    return dataset_view


def test_frame_discontinuity(image_dataset_view):
    sequence_ids = image_dataset_view.values("id")

    pair_view = image_dataset_view.select([sequence_ids[0], sequence_ids[1]])
    ctx = {
        "dataset": pair_view._dataset,
        "view": pair_view,
        "params": {
            "input_annotation_field": "labels_test",
            "output_annotation_field": "labels_test_propagated",
            "propagation_method": "sam2",
            "sort_field": "frame_number",
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    assert result.result["message"] == "Annotations propagated from labels_test to labels_test_propagated", "Labels were not propagated correctly"  # type: ignore[index]

    # TODO(neeraja): next...
    # image_a = first frame
    # image_c = furthest frame that does not break sam2
    # image_d = first frame that breaks sam2
    # find params such that (image_a, image_c) is continuous and (image_a, image_d) is discontinuous


# def test_frame_discontinuity_temp():
    # dataset = foz.load_zoo_dataset(
    #     "https://github.com/voxel51/davis-2017",
    #     split="validation",
    #     format="image",
    # )
    # sequence_ids = dataset.values("id")
    # first_sample_id = sequence_ids[0]  # type: ignore
    # last_sample_id = sequence_ids[-1]  # type: ignore
    # pair_view = dataset.select([first_sample_id, last_sample_id])
    # first_sample = pair_view.first()
    # first_sample["labels_test"] = first_sample["ground_truth"]
    # first_sample.save()
    # last_sample = pair_view.last()
    # last_sample["labels_test"] = last_sample["ground_truth"]
    # last_sample.save()

    # ctx = {
    #     "dataset": pair_view._dataset,
    #     "view": pair_view,
    #     "params": {
    #         "input_annotation_field": "labels_test",
    #         "output_annotation_field": "labels_test_propagated",
    #         "propagation_method": "sam2",
    #         "sort_field": "frame_number",
    #     },
    # }

    # result = foo.execute_operator(
    #     "@51labs/label_propagation/propagate_labels", ctx
    # )
    # print(result.result["message"])  # type: ignore[index]

    # breakpoint()

    # if result.result["message"] != "Annotations propagated from labels_test to labels_test_propagated":  # type: ignore[index]
    #     breakpoint()