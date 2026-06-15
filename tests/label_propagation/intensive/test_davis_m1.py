import pytest
import numpy as np

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


@pytest.fixture
def partially_labeled_grouped_dataset_view(
    partially_labeled_image_dataset_view,
):
    grouped_dataset_view = partially_labeled_image_dataset_view.group_by(
        "sequence_id", order_by="frame_number"
    )
    return grouped_dataset_view


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
    dataset_view = dataset_view.match_frames(F("frame_number") <= 20)
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


@pytest.mark.parametrize(
    "partially_labeled_view_fixture",
    [
        "partially_labeled_image_dataset_view",
        "partially_labeled_grouped_dataset_view",
    ],
)
def test_propagate_labels_m1_image(request, partially_labeled_view_fixture):
    partially_labeled_view = request.getfixturevalue(
        partially_labeled_view_fixture
    )
    check_view = (
        partially_labeled_view.flatten()
        if partially_labeled_view.media_type == "group"
        else partially_labeled_view
    )
    n = len(check_view.sort_by("new_frame_number"))

    ctx = {
        "dataset": partially_labeled_view._dataset,
        "view": partially_labeled_view,
        "params": {
            "annotation_field": "labels_test_m1",
            "label_index": 0,
            "start_frame_number": 1,
            "end_frame_number": n,
            "sort_field": "new_frame_number",
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels_m1", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    detection_area = (
        lambda det: (det.bounding_box[2] * det.bounding_box[3])
        if det.bounding_box is not None
        else 0
    )
    areas = [
        sum([detection_area(det) for det in prop])
        for prop in check_view.values("labels_test_m1.detections")
    ]

    assert np.min(areas) > 0.05

    sorted_view = check_view.sort_by("new_frame_number")
    seed_positions = {n // 3, n * 2 // 3}
    output_indices = sorted_view.values("labels_test_m1.detections.index")

    frames_with_nonzero_index_labels = 0
    for pos, frame_indices in enumerate(output_indices):
        if frame_indices is None:
            continue
        if pos in seed_positions:
            frames_with_nonzero_index_labels += int(
                any(idx != 0 for idx in frame_indices)
            )
            assert 0 in frame_indices
        else:
            assert set(frame_indices) <= {0}

    assert frames_with_nonzero_index_labels == len(seed_positions)


def test_propagate_labels_m1_video(partially_labeled_video_dataset_view):
    n = len(partially_labeled_video_dataset_view.first().frames)

    ctx = {
        "dataset": partially_labeled_video_dataset_view._dataset,
        "view": partially_labeled_video_dataset_view,
        "params": {
            "annotation_field": "frames.labels_test_m1",
            "label_index": 0,
            "start_frame_number": 1,
            "end_frame_number": n,
        },
    }
    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels_m1", ctx
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
            "frames.labels_test_m1.detections"
        )
    ]
    assert np.min(areas) > 0.05

    all_indices = partially_labeled_video_dataset_view.values(
        "frames.labels_test_m1.detections.index"
    )
    for sample, sample_indices in zip(
        partially_labeled_video_dataset_view, all_indices
    ):
        frame_numbers = sorted(sample.frames.keys())
        n_frames = len(frame_numbers)
        seed_positions = {n_frames // 3, n_frames * 2 // 3}
        seed_keys = {frame_numbers[i] for i in seed_positions}

        frames_with_nonzero_index_labels = 0
        for fn, frame_indices in zip(frame_numbers, sample_indices):
            if frame_indices is None:
                continue
            if fn in seed_keys:
                frames_with_nonzero_index_labels += int(
                    any(idx != 0 for idx in frame_indices)
                )
                assert 0 in frame_indices
            else:
                assert set(frame_indices) <= {0}

        assert frames_with_nonzero_index_labels == len(seed_positions)
        assert (
            len(set(sample_indices[0]).intersection(set(sample_indices[-1])))
            > 0
        )

    all_instances = partially_labeled_video_dataset_view.values(
        "frames.labels_test_m1.detections.instance"
    )
    for sample_instances in all_instances:
        instance_ids_first_frame = [
            str(instance["_id"]) for instance in sample_instances[0]
        ]
        instance_ids_last_frame = [
            str(instance["_id"]) for instance in sample_instances[-1]
        ]
        assert (
            len(
                set(instance_ids_first_frame).intersection(
                    set(instance_ids_last_frame)
                )
            )
            > 0
        )
