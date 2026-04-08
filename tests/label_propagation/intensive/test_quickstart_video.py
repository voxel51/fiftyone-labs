import pytest
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import cv2

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

_TEST_PKG_DIR = Path(__file__).resolve().parent.parent
PLUGINS_DIR = _TEST_PKG_DIR.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from label_propagation.suc_utils import evaluate  # type: ignore


@pytest.fixture
def video_dataset_view():
    dataset = foz.load_zoo_dataset("quickstart-video").limit(1)
    view = dataset.match_frames(F("frame_number") < 20)
    return view


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
    for sample in video_dataset_view.iter_samples(autosave=True):
        for frame_number, frame in sample.frames.items():
            if frame_number == 1:
                frame["labels_test"] = frame["detections"]
            else:
                frame["labels_test"] = fo.Detections(detections=[])

    return video_dataset_view


def test_propagate_labels_video(partially_labeled_video_dataset_view):
    ctx = {
        "dataset": partially_labeled_video_dataset_view._dataset,
        "view": partially_labeled_video_dataset_view,
        "params": {
            "input_annotation_field": "frames.labels_test",
            "output_annotation_field": "frames.labels_test_propagated",
            "propagation_method": "sam2",
            "sort_field": "frames.frame_number",
            "max_batch_size": 32,
        },
    }
    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    pred_detections = partially_labeled_video_dataset_view.values(
        "frames.labels_test_propagated"
    )
    gt_detections = partially_labeled_video_dataset_view.values(
        "frames.detections"
    )

    scores = []
    for sample_pred_detections, sample_gt_detections in zip(
        pred_detections, gt_detections
    ):
        video_path = partially_labeled_video_dataset_view.first()["filepath"]
        video = cv2.VideoCapture(video_path)
        WW, HH = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_idx = 0
        for pred, gt in zip(sample_pred_detections, sample_gt_detections):
            scores.append(evaluate(pred, gt))
            frame_idx += 1
            # save an image with the predictions and the ground truth
            _, image = video.read()
            for bbox in pred.detections:
                x, y, w, h = bbox.bounding_box
                cv2.rectangle(image, (int(x * WW), int(y * HH)), (int((x + w) * WW), int((y + h) * HH)), (255, 0, 0))
                if bbox.mask is not None:
                    y1, y2 = int(y * HH), int((y + h) * HH)
                    x1, x2 = int(x * WW), int((x + w) * WW)
                    roi = image[y1:y2, x1:x2]
                    m = np.asarray(bbox.mask)
                    if m.ndim == 3:
                        m = m[..., 0] if m.shape[-1] == 1 else m.max(axis=-1)
                    m = (
                        m * 255 if float(m.max()) <= 1 else np.clip(m, 0, 255)
                    ).astype(np.uint8)
                    m = cv2.resize(m, (roi.shape[1], roi.shape[0]), cv2.INTER_NEAREST)
                    blue = np.zeros_like(roi)
                    blue[:, :, 0] = m
                    cv2.addWeighted(roi.copy(), 0.5, blue, 0.5, 0, image[y1:y2, x1:x2])
            for bbox in gt.detections:
                x, y, w, h = bbox.bounding_box
                cv2.rectangle(image, (int(x * WW), int(y * HH)), (int((x + w) * WW), int((y + h) * HH)), (0, 255, 0))
                cv2.imwrite(f"pred_{frame_idx}.png", image)

    print("per frame scores: ", scores)
    assert np.min(scores) > 0.7

    all_indices = partially_labeled_video_dataset_view.values(
        "frames.labels_test_propagated.detections.index"
    )
    for sample_indices in all_indices:
        assert (
            len(set(sample_indices[0]).intersection(set(sample_indices[-1])))
            > 0
        )

    all_instances = partially_labeled_video_dataset_view.values(
        "frames.labels_test_propagated.detections.instance"
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
