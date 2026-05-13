import pytest
import numpy as np
from pathlib import Path
import sys

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

_TEST_PKG_DIR = Path(__file__).resolve().parent.parent
PLUGINS_DIR = _TEST_PKG_DIR.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from label_propagation.suc_utils import (  # type: ignore
    evaluate_detections,
)

PROPAGATION_METHOD = ""

@pytest.fixture(params=[0, 2, 4, 6, 8, 10, 12])
def image_dataset_view(request):
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/mose-v2",
        split="train",
    )
    sequence_ids = sorted(dataset.distinct("sequence_id"))  # type: ignore[arg-type]
    sequence_id = sequence_ids[request.param]
    view = dataset.match(F("sequence_id") == sequence_id)
    # view = view.match(F("frame_number") < 9)
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
    sequence_id = view.first()["sequence_id"]

    ctx = {
        "dataset": view._dataset,
        "view": view,
        "params": {
            "input_annotation_field": "labels_test",
            "output_annotation_field": "labels_test_propagated",
            "propagation_method": PROPAGATION_METHOD,
            "sort_field": "new_frame_number",
        },
    }

    result = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    print(result.result["message"])  # type: ignore[index]

    scores_eval_detections = evaluate_detections(
        view,
        pred_field="labels_test_propagated",
        gt_field="ground_truth",
    )

    with open(f"scores_mose_{sequence_id}_{PROPAGATION_METHOD}.csv", "w") as f:
        for i, score in enumerate(scores_eval_detections):
            f.write(f"{i},{score}\n")
