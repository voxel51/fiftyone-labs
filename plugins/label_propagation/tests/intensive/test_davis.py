import pytest
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

# from ..utils import evaluate


@pytest.fixture
def dataset_view():
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    SELECT_SEQUENCES = ["bike-packing"]
    dataset_view = dataset.match_tags(SELECT_SEQUENCES)
    dataset_view = dataset_view.match(F("frame_number").to_int() < 9)
    return dataset_view


@pytest.fixture
def partially_labeled_dataset_view(dataset_view):
    if "labels_test" in dataset_view._dataset.get_field_schema():
        try:
            dataset_view._dataset.delete_sample_field(
                "labels_test", error_level=2
            )
        except AttributeError:
            assert (
                "labels_test" not in dataset_view._dataset.get_field_schema()
            ), "Unable to delete labels_test field"

        dataset_view._dataset.add_sample_field(
            "labels_test",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

    sequences = dataset_view.distinct("tags")
    sequences.remove("val")
    new_frame_number = 0
    for seq in sequences:
        seq_slice = dataset_view.match_tags(seq).sort_by("frame_number")
        seq_slice.set_values(
            "new_frame_number",
            [new_frame_number + ii for ii in range(len(seq_slice))],
        )
        new_frame_number += len(seq_slice)

        # label only the first
        # TODO(neeraja): support backward propagation [in a follow-up PR]
        exemplar_sample = seq_slice.first()
        exemplar_sample["labels_test"] = exemplar_sample["ground_truth"]
        exemplar_sample.save()

    return dataset_view


def test_assign_exemplar_frames(dataset_view):
    ctx = {
        "dataset": dataset_view._dataset,
        "view": dataset_view,
        "params": {
            "method": "heuristic",
            "sort_field": "frame_number",
            "exemplar_frame_field": "exemplar_test",
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/assign_exemplar_frames", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    exemplars = dataset_view.values("exemplar_test.is_exemplar")
    assert exemplars[0]
    assert np.mean(exemplars) < 0.33


def test_propagate_labels(partially_labeled_dataset_view):
    ctx = {
        "dataset": partially_labeled_dataset_view._dataset,
        "view": partially_labeled_dataset_view,
        "params": {
            "input_annotation_field": "labels_test",
            "output_annotation_field": "labels_test_propagated",
            "sort_field": "frame_number",
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    detection_area = lambda det: (det.bounding_box[2] * det.bounding_box[3]) if det.bounding_box is not None else 0
    areas = [sum([detection_area(det) for det in prop]) for prop in partially_labeled_dataset_view.values("labels_test_propagated.detections")]
    assert np.min(areas) > 0.35

    # TODO(neeraja): add evaluation [in a follow-up PR]
