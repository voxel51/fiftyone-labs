"""
Run this script in the foe environment
"""

import pytest
import numpy as np

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.core.storage as fos
import fiftyone.zoo as foz
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

    cloud_dir = "gs://voxel51-test/DAVIS-2017/"
    rel_dir = "/".join(dataset.first()["filepath"].split("/")[:-2])

    fos.upload_media(
        dataset_view,
        cloud_dir,
        rel_dir=rel_dir,
        update_filepaths=True,  # reflects new paths
        overwrite=False,  # but doesn't overwrite cloud media
        progress=True,
    )
    return dataset_view


@pytest.fixture
def video_dataset_view():
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        dataset_name="davis-2017-video-validation",
        split="validation",
        format="video",
    )
    SELECT_SEQUENCES = ["bike-packing", "bmx-trees"]
    dataset_view = dataset.match_tags(SELECT_SEQUENCES)
    dataset_view.match_frames(F("frame_number") <= 4).keep_frames()

    cloud_dir = "gs://voxel51-test/DAVIS-2017/"
    rel_dir = "/".join(dataset.first()["filepath"].split("/")[:-2])

    fos.upload_media(
        dataset_view,
        cloud_dir,
        rel_dir=rel_dir,
        update_filepaths=True,  # reflects new paths
        overwrite=False,  # but doesn't overwrite cloud media
        progress=True,
    )
    return dataset_view


@pytest.fixture
def partially_labeled_image_dataset_view(image_dataset_view):
    if "labels_test" in image_dataset_view._dataset.get_field_schema():
        image_dataset_view._dataset.delete_sample_field(
            "labels_test", error_level=2
        )
    if (
        "labels_test_propagated"
        in image_dataset_view._dataset.get_field_schema()
    ):
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
def partially_labeled_video_dataset_view(video_dataset_view):
    if (
        "labels_test"
        not in video_dataset_view._dataset.get_frame_field_schema()
    ):
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
        .set_field("frames.labels_test", F("ground_truth"))
        .save()
    )

    return video_dataset_view


def test_propagate_labels_image(partially_labeled_image_dataset_view):
    ctx = {
        "dataset": partially_labeled_image_dataset_view._dataset,
        "view": partially_labeled_image_dataset_view,
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
        for prop in partially_labeled_image_dataset_view.values(
            "labels_test_propagated.detections"
        )
    ]
    assert np.min(areas) > 0.35

    # TODO(neeraja): add evaluation [in a follow-up PR]

    indices = partially_labeled_image_dataset_view.values(
        "labels_test_propagated.detections.index"
    )
    assert (
        indices[0] == indices[-1]
    )  # same number of objects in the first and last frames
