import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.zoo as foz
from fiftyone.core.expressions import ViewField as F

from sam2.sam2_video_predictor import SAM2VideoPredictor

_TEST_PKG_DIR = Path(__file__).resolve().parent.parent
PLUGINS_DIR = _TEST_PKG_DIR.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from label_propagation.suc_utils import evaluate  # type: ignore

# Same checkpoint/config as FiftyOne zoo `segment-anything-2-hiera-tiny-video-torch`
# (sam2_hiera_t.yaml + sam2_hiera_tiny.pt); required id for `from_pretrained`.
MODEL_ID = "facebook/sam2-hiera-tiny"

# `match_frames` only filters FiftyOne; SAM2 still reads the whole file unless we cap
# JPEG extraction (see `_mp4_to_jpg_dir`) and only write back this many frames.
MAX_FRAMES = 20


@pytest.fixture
def video_dataset_view():
    dataset = foz.load_zoo_dataset("quickstart-video").limit(1)
    view = dataset.match_frames(F("frame_number") <= MAX_FRAMES)
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
        frame_keys = sorted(sample.frames.keys())[:MAX_FRAMES]
        for frame_number in frame_keys:
            frame = sample.frames[frame_number]
            if frame_number == 1:
                frame["labels_test"] = frame["detections"]
            else:
                frame["labels_test"] = fo.Detections(detections=[])

    return video_dataset_view


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


def _mp4_to_jpg_dir(mp4_path: str, max_frames: int | None = None) -> str:
    """SAM2 loads MP4 via `decord`; JPEG folders use PIL only (works without decord)."""
    d = tempfile.mkdtemp(prefix="sam2_frames_")
    cap = cv2.VideoCapture(mp4_path)
    idx = 0
    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(
                os.path.join(d, f"{idx:05d}.jpg"), quality=95
            )
            idx += 1
    finally:
        cap.release()
    if idx == 0:
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError(f"no frames read from {mp4_path}")
    return d


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


def test_propagate_labels_video_sam2(partially_labeled_video_dataset_view):
    view = partially_labeled_video_dataset_view
    sample = view.first()
    video_path = sample.filepath
    ordered_fn = sorted(sample.frames.keys())[:MAX_FRAMES]
    prompt_fn = ordered_fn[0]
    prompt_sam_idx = 0

    src = sample.frames[prompt_fn].labels_test
    assert src.detections

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(MODEL_ID, device=device)
    frame_dir = _mp4_to_jpg_dir(video_path, max_frames=MAX_FRAMES)
    by_sam_idx = {}
    num_sam = 0
    try:
        state = predictor.init_state(frame_dir)
        W, H = int(state["video_width"]), int(state["video_height"])

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
                frame_idx=prompt_sam_idx,
                obj_id=oid,
                box=box,
            )

        for out_idx, out_obj_ids, video_res_masks in predictor.propagate_in_video(
            state
        ):
            by_sam_idx[out_idx] = _video_masks_to_detections(
                video_res_masks, out_obj_ids, id_to_label, W, H
            )

        num_sam = state["num_frames"]
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    for sam_i in range(num_sam):
        if sam_i >= len(ordered_fn):
            break
        fn = ordered_fn[sam_i]
        sample.frames[fn]["labels_test"] = by_sam_idx[sam_i]

    sample.save()

    pred_detections = view.values("frames.labels_test")
    gt_detections = view.values("frames.detections")
    scores = []
    for sample_pred_detections, sample_gt_detections in zip(
        pred_detections, gt_detections
    ):
        video_path_eval = view.first()["filepath"]
        video = cv2.VideoCapture(video_path_eval)
        WW, HH = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        frame_idx = 0
        for pred, gt in zip(sample_pred_detections, sample_gt_detections):
            scores.append(evaluate(pred, gt))
            frame_idx += 1
            _, image = video.read()
            for bbox in pred.detections:
                x, y, w, h = bbox.bounding_box
                cv2.rectangle(
                    image,
                    (int(x * WW), int(y * HH)),
                    (int((x + w) * WW), int((y + h) * HH)),
                    (255, 0, 0),
                )
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
                    cv2.addWeighted(
                        roi.copy(), 0.5, blue, 0.5, 0, image[y1:y2, x1:x2]
                    )
            for bbox in gt.detections:
                x, y, w, h = bbox.bounding_box
                cv2.rectangle(
                    image,
                    (int(x * WW), int(y * HH)),
                    (int((x + w) * WW), int((y + h) * HH)),
                    (0, 255, 0),
                )
                cv2.imwrite(f"pred_standalone_{frame_idx}.png", image)

    print("per frame scores: ", scores)
    assert np.min(scores) > 0.7

    idx_lists = view.values("frames.labels_test.detections.index")
    first, last = idx_lists[0][0], idx_lists[0][-1]
    assert len(set(first).intersection(set(last))) > 0
