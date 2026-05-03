"""Unit tests for functions in the label_propagation plugin."""

from __future__ import annotations

import pytest
from pathlib import Path
import sys
import numpy as np
import cv2
import tempfile
import shutil
from PIL import Image
import os

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz
import fiftyone.core.labels as fol
from fiftyone.core.expressions import ViewField as F

_TEST_PKG_DIR = Path(__file__).resolve().parent
PLUGINS_DIR = _TEST_PKG_DIR.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

# Same checkpoint/config as FiftyOne zoo `segment-anything-2-hiera-tiny-video-torch`
# (sam2_hiera_t.yaml + sam2_hiera_tiny.pt); required id for `from_pretrained`.
MODEL_ID = "facebook/sam2-hiera-tiny"


@pytest.fixture(params=["dance-twirl", "motocross-jump", "scooter-black"])
def image_dataset_view(request):
    sequence = request.param
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    dataset_view = dataset.match_tags([sequence]).sort_by("frame_number")

    for ii, sample in enumerate(dataset_view.iter_samples()):
        if ii in [0, 2]:
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



@pytest.fixture(params=["motocross-jump"])
def video_dataset_view(request):
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        dataset_name="davis-2017-video-validation",
        split="validation",
        format="video",
    )
    sequence_id = request.param
    dataset_view = dataset.match(F("sequence_id") == sequence_id)

    ds = dataset_view._dataset
    if "labels_test" not in ds.get_frame_field_schema():
        ds.add_frame_field(
            "labels_test",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )
    if "labels_test_propagated_native" not in ds.get_frame_field_schema():
        ds.add_frame_field(
            "labels_test_propagated_native",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )
    for sample in dataset_view.iter_samples(autosave=True):
        for frame_number, frame in sample.frames.items():
            if frame_number == 1:
                frame["labels_test"] = frame["ground_truth"]
            else:
                frame["labels_test"] = fo.Detections(detections=[])
    return dataset_view


### utility functions for SAM2 native inference ###

def _mask_to_detection(mask_bool: np.ndarray, W: int, H: int, label: str, index: int):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bb = [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]
    crop = mask_bool[y1:y2, x1:x2]
    return fol.Detection(
        label=label, bounding_box=bb, mask=crop.astype(np.uint8), index=index
    )


def _video_masks_to_detections(video_res_masks, obj_ids, id_to_label, W, H):
    vm = video_res_masks
    if vm.dim() == 4:
        det_list = []
        for i, oid in enumerate(obj_ids):
            mask = (vm[i, 0] > 0.0).cpu().numpy()
            det = _mask_to_detection(mask, W, H, id_to_label[oid], int(oid))
            if det is not None:
                det_list.append(det)
        return fol.Detections(detections=det_list)
    raise AssertionError(f"unexpected mask tensor shape {tuple(vm.shape)}")


def sam2_native(sample):
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    video_path = sample.filepath
    exemplar_fn = 1
    src = sample.frames[exemplar_fn].labels_test
    assert src.detections

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(MODEL_ID, device=device)

    state = predictor.init_state(video_path)
    W, H = int(state["video_width"]), int(state["video_height"])

    # SAM2 uses 0-based frame indices; FiftyOne DAVIS frames use 1-based keys.
    sam2_prompt_idx = exemplar_fn - 1

    id_to_label = {}
    for i, det in enumerate(src.detections):
        oid = det.index if det.index is not None else i + 1
        id_to_label[oid] = det.label
        x, y, w, h = det.bounding_box
        box = np.array(
            [x * W, y * H, (x + w) * W, (y + h) * H], dtype=np.float32
        )
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=sam2_prompt_idx,
            obj_id=oid,
            box=box,
        )

    for out_idx, out_obj_ids, video_res_masks in predictor.propagate_in_video(
        state
    ):
        fo_fn = out_idx + 1
        if fo_fn not in sample.frames:
            continue
        sample.frames[fo_fn]["labels_test_propagated_native"] = (
            _video_masks_to_detections(
                video_res_masks, out_obj_ids, id_to_label, W, H
            )
        )

    return sample

### end of utility functions for SAM2 native inference ###


def test_sam2_parity(video_dataset_view):
    view = video_dataset_view
    ctx = {
        "dataset": view._dataset,
        "view": view,
        "params": {
            "input_annotation_field": "frames.labels_test",
            "output_annotation_field": "frames.labels_test_propagated",
            "propagation_method": "sam2",
            "sort_field": "frames.frame_number",
        },
    }

    # Run native SAM2 before the plugin: ``propagate_labels`` monkey-patches
    # ``sam2.utils.misc.load_video_frames`` to accept (sample, reader) tuples
    for _ in view.map_samples(
        sam2_native,
        save=True,
        num_workers=1,
        parallelize_method="thread",
    ):
        pass

    # TODO(neeraja): the monkey patch doesn't load correctly here: fix this.
    result_plugin = foo.execute_operator(
        "@51labs/label_propagation/propagate_labels", ctx
    )
    print(result_plugin.result["message"])  # type: ignore[index]

    for sample in view.iter_samples():
        for _, frame in sample.frames.items():
            det1 = frame["labels_test_propagated_native"]
            det2 = frame["labels_test_propagated"]

            assert det1 is not None and det2 is not None
            assert len(det1.detections) == len(det2.detections)
            for det1_det, det2_det in zip(
                sorted(det1.detections, key=lambda x: x.index), sorted(det2.detections, key=lambda x: x.index)
            ):
                assert det1_det.label == det2_det.label
                for boxval1, boxval2 in zip(det1_det.bounding_box, det2_det.bounding_box):
                    assert abs(boxval1 - boxval2) < 1e-6
