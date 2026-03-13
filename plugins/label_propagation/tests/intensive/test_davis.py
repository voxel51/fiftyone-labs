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


@pytest.fixture
def image_dataset_view():
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
def partially_labeled_image_dataset_view(image_dataset_view):
    if "labels_test" in image_dataset_view._dataset.get_field_schema():
        image_dataset_view._dataset.delete_sample_field(
            "labels_test", error_level=2
        )
    if "labels_test_propagated" in image_dataset_view._dataset.get_field_schema():
        image_dataset_view._dataset.delete_sample_field(
            "labels_test_propagated", error_level=2
        )

    if "labels_test" not in image_dataset_view._dataset.get_field_schema():
        image_dataset_view._dataset.add_sample_field(
            "labels_test",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

    sequences = image_dataset_view.distinct("tags")
    sequences.remove("val")
    new_frame_number = 0
    for seq in sequences:
        seq_slice = image_dataset_view.match_tags(seq).sort_by("frame_number")
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

    return image_dataset_view


@pytest.fixture
def partially_labeled_grouped_dataset_view(partially_labeled_image_dataset_view):
    grouped_dataset_view = partially_labeled_image_dataset_view.group_by("sequence_id", order_by="frame_number")
    return grouped_dataset_view


@pytest.fixture
def video_dataset_view():
    dataset = foz.load_zoo_dataset("quickstart-video").limit(2)
    dataset.match_frames(F("frame_number") <= 4).keep_frames()
    return dataset


@pytest.fixture
def partially_labeled_video_dataset_view(video_dataset_view):
    if "labels_test" not in video_dataset_view._dataset.get_frame_field_schema():
        video_dataset_view._dataset.add_frame_field(
            "labels_test",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )
    (
        video_dataset_view.match_frames(F("frame_number") > 1)
        .set_field("frames.labels_test", fo.Detections(detections=[]))
        .save()
    )
    (
        video_dataset_view.match_frames(F("frame_number") == 1)
        .set_field("frames.labels_test", F("detections"))
        .save()
    )

    return video_dataset_view


@pytest.mark.parametrize(
    "view_fixture", [
        pytest.param(
            "partially_labeled_image_dataset_view",
            marks=pytest.mark.dependency(),
        ),
        pytest.param(
            "partially_labeled_grouped_dataset_view",
            marks=pytest.mark.dependency(),
        ),
    ],
)
def test_temporal_segmentation(request, view_fixture):
    view = request.getfixturevalue(view_fixture)
    ctx = {
        "dataset": view._dataset,
        "view": view,
        "params": {
            "temporal_segmentation_method": "heuristic",
            "temporal_segments_field": "temporal_segments_test",
            "sort_field": "new_frame_number",
        },
    }
    result = foo.execute_operator(
        "@51labs/label_propagation/temporal_segmentation", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    classifications = view.values("temporal_segments_test")
    assert all(cc is not None and cc.classifications for cc in classifications)

    labels = view.values("temporal_segments_test.classifications.label")
    assert all(len(ll) == 1 for ll in labels)
    assert len(set(np.array(labels).flatten())) == 1

    exemplar_scores = view.values("temporal_segments_test.classifications.exemplar_score")
    assert set(np.array(exemplar_scores).flatten()) == {0}


@pytest.mark.parametrize(
    "view_fixture", [
        pytest.param(
            "partially_labeled_image_dataset_view",
            marks=pytest.mark.dependency(
                depends=["test_temporal_segmentation[partially_labeled_image_dataset_view]"]
            ),
        ),
        pytest.param(
            "partially_labeled_grouped_dataset_view",
            marks=pytest.mark.dependency(
                depends=["test_temporal_segmentation[partially_labeled_grouped_dataset_view]"]
            ),
        ),
    ],
)
def test_temporal_segment_exemplar_scoring(request, view_fixture):
    view = request.getfixturevalue(view_fixture)
    ctx = {
        "dataset": view._dataset,
        "view": view,
        "params": {
            "exemplar_scoring_method": "first_frame",
            "temporal_segments_field": "temporal_segments_test",
            "sort_field": "new_frame_number",
        },
    }
    result = foo.execute_operator(
        "@51labs/label_propagation/select_exemplars", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    temporal_classifications = view.values("temporal_segments_test")
    exemplar_scores = [
        getattr(c.classifications[0], "exemplar_score", 0)
        for c in temporal_classifications
        if c and c.classifications
    ]
    assert np.abs(np.mean(exemplar_scores) - 1.0/len(temporal_classifications)) < 1e-6


def test_temporal_segmentation_video(partially_labeled_video_dataset_view):
    view = partially_labeled_video_dataset_view
    ctx = {
        "dataset": view._dataset,
        "view": view,
        "params": {
            "temporal_segmentation_method": "heuristic",
            "temporal_segments_field": "temporal_segments_test",
            "sort_field": "frames.frame_number",
        },
    }
    result = foo.execute_operator(
        "@51labs/label_propagation/temporal_segmentation", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    temporal_classifications = view.values("temporal_segments_test")
    assert all(
        len(temporal_classifications[ii].detections) == 1
        for ii in range(len(view))
    )


@pytest.mark.parametrize(
    "partially_labeled_view_fixture",
    ["partially_labeled_image_dataset_view", "partially_labeled_grouped_dataset_view"],
)
def test_propagate_labels_image(request, partially_labeled_view_fixture):
    partially_labeled_view = request.getfixturevalue(partially_labeled_view_fixture)
    ctx = {
        "dataset": partially_labeled_view._dataset,
        "view": partially_labeled_view,
        "params": {
            "input_annotation_field": "labels_test",
            "output_annotation_field": "labels_test_propagated",
            "propagation_method": "sam2",
            "sort_field": "new_frame_number",
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    detection_area = (
        lambda det: (det.bounding_box[2] * det.bounding_box[3])
        if det.bounding_box is not None
        else 0
    )
    areas = [
        sum([detection_area(det) for det in prop])
        for prop in partially_labeled_view.values(
            "labels_test_propagated.detections"
        )
    ]
    assert np.min(areas) > 0.35

    # TODO(neeraja): add evaluation [in a follow-up PR]

    indices = partially_labeled_view.values(
        "labels_test_propagated.detections.index"
    )
    assert indices[0] == indices[-1]  # same number of objects in the first and last frames


def test_propagate_labels_video(partially_labeled_video_dataset_view):
    ctx = {
        "dataset": partially_labeled_video_dataset_view._dataset,
        "view": partially_labeled_video_dataset_view,
        "params": {
            "input_annotation_field": "frames.labels_test",
            "output_annotation_field": "frames.labels_test_propagated",
            "propagation_method": "sam2",
            "sort_field": "frames.frame_number",
        },
    }
    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    detection_area = (
        lambda det: (det.bounding_box[2] * det.bounding_box[3])
        if det.bounding_box is not None
        else 0
    )
    areas = [
        sum(
            detection_area(det)
            for frame_detections in sample_detections
            for det in frame_detections
        )
        for sample_detections in partially_labeled_video_dataset_view.values(
            "frames.labels_test_propagated.detections"
        )
    ]
    assert np.min(areas) > 0.1

    # TODO(neeraja): add evaluation [in a follow-up PR]
