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
def dataset():
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    SELECT_SEQUENCES = ["bike-packing"]
    # SELECT_SEQUENCES = ["bike-packing", "car-roundabout"]
    # TODO(neeraja): support multiple sequences
    dataset = dataset.match_tags(SELECT_SEQUENCES)
    dataset = dataset.match(F("frame_number").to_int() < 9)
    return dataset


@pytest.fixture
def partially_labeled_dataset(dataset):
    if "labels_test" in dataset._dataset.get_field_schema():
        dataset._dataset.delete_sample_field("labels_test")
        dataset._dataset.add_sample_field(
            "labels_test",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

    sequences = dataset.distinct("tags")
    sequences.remove("val")
    new_frame_number = 0
    for seq in sequences:
        dataset_slice = dataset.match_tags(seq).sort_by("frame_number")
        dataset_slice.set_values(
            "new_frame_number",
            [new_frame_number + ii for ii in range(len(dataset_slice))],
        )
        new_frame_number += len(dataset_slice)

        # label only the first
        exemplar_sample = dataset_slice.first()
        exemplar_sample["labels_test"] = exemplar_sample["ground_truth"]
        exemplar_sample.save()

        # TODO(neeraja): support labeling an arbitrary frame

    return dataset


def test_propagate_labels(partially_labeled_dataset):
    ctx2 = {
        "dataset": partially_labeled_dataset._dataset,
        "view": partially_labeled_dataset,
        "params": {
            "input_annotation_field": "labels_test",
            "output_annotation_field": "labels_test_propagated",
            "sort_field": "frame_number",
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx2
    )
    print(result.result["message"])  # type: ignore[index]

    # TODO(neeraja): add evaluation
