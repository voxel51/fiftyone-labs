"""Unit tests for functions in the label_propagation plugin."""

from __future__ import annotations

import pytest
from pathlib import Path
import sys
from typing import Tuple
import numpy as np
import cv2

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz
from fiftyone.core.expressions import ViewField as F


@pytest.fixture(params=[
    "dance-twirl", "motocross-jump", "scooter-black"
])
def image_dataset_view(request):
    sequence = request.param
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    dataset_view = dataset.match_tags([sequence]).sort_by("frame_number")

    if "labels_test" in dataset_view._dataset.get_field_schema():
        dataset_view._dataset.delete_sample_field(
            "labels_test", error_level=2
        )
    dataset_view._dataset.add_sample_field(
        "labels_test",
        fo.EmbeddedDocumentField,
        embedded_doc_type=fo.Detections,
    )

    for ii, sample in enumerate(dataset_view.iter_samples()):
        if ii % 2 == 0:
            sample["labels_test"] = sample["ground_truth"]
            sample.save()

    return dataset_view


def test_sam2_dtype_handling(image_dataset_view):
    sequence_ids = image_dataset_view.values("id")
    three_frame_view = image_dataset_view.select(
        [sequence_ids[0], sequence_ids[1], sequence_ids[2]]
    )
    ctx = {
        "dataset": three_frame_view._dataset,
        "view": three_frame_view,
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
