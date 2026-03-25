import pytest
import numpy as np
import cv2
from PIL import Image

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

from ...suc_utils import evaluate_matched


def _segmentation_to_detections(
    segmentation: fo.Segmentation,
) -> fo.Detections:
    """Convert an indexed-PNG fo.Segmentation to fo.Detections.

    MOSE annotation masks are 8-bit indexed PNGs where pixel value = object
    instance ID (0 = background). This mirrors what the DAVIS loader produces
    natively, so the propagation operator sees the same input format.
    """
    mask = np.array(Image.open(segmentation.mask_path))  # type: ignore[arg-type]
    h, w = mask.shape

    detections = []
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue  # background
        binary = (mask == obj_id).astype(np.uint8)
        x, y, bw, bh = cv2.boundingRect(binary)
        if bw == 0 or bh == 0:
            continue
        detections.append(
            fo.Detection(
                label=str(obj_id),
                bounding_box=[x / w, y / h, bw / w, bh / h],
                mask=binary[y : y + bh, x : x + bw],
            )
        )
    return fo.Detections(detections=detections)


@pytest.fixture(params=[0, 2, 4, 6, 8])
def image_dataset_view(request):
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/mose-v2",
        split="train",
    )
    sequence_ids = sorted(dataset.distinct("sequence_id"))  # type: ignore[arg-type]
    sequence_id = sequence_ids[request.param]
    view = dataset.match(F("sequence_id") == sequence_id)
    view = view.match(F("frame_number") < 9)
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
        seq_slice.set_values(
            "new_frame_number",
            [new_frame_number + ii for ii in range(len(seq_slice))],
        )
        new_frame_number += len(seq_slice)

        # label only the first frame
        exemplar_sample = seq_slice.first()
        exemplar_sample["labels_test"] = _segmentation_to_detections(
            exemplar_sample["ground_truth"]
        )
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
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    pred_detections = view.values("labels_test_propagated")
    gt_detections = view.values("ground_truth")
    gt_detections = [_segmentation_to_detections(gt) for gt in gt_detections]

    scores = []
    for pred, gt in zip(pred_detections, gt_detections):
        scores.append(evaluate_matched(pred, gt))

    print("per frame scores: ", scores)
    assert np.mean(scores) > 0.6

    indices = view.values("labels_test_propagated.detections.index")
    assert (
        indices[0] == indices[-1]
    )  # same number of objects in the first and last frames

    indices = view.values("labels_test_propagated.detections.index")
    assert (
        len(set(indices[0]).intersection(set(indices[-1]))) > 0
    )  # similar number of objects in the first and last frames
