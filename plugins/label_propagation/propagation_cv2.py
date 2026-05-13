import logging
from typing import Union, Optional
import numpy as np
import cv2

import fiftyone as fo
import fiftyone.core.dataset as fod

from .suc_utils import normalized_bbox_to_pixel_coords

logger = logging.getLogger(__name__)

CV2_PROPAGATION_METHODS = ["cv2_csrt", "cv2_kcf", "cv2_medianflow", "cv2_mosse"]


def _make_tracker(method: str):
    if method == "cv2_csrt":
        return cv2.TrackerCSRT_create()
    if method == "cv2_kcf":
        return cv2.TrackerKCF_create()
    if method == "cv2_medianflow":
        return cv2.legacy.TrackerMedianFlow_create()
    if method == "cv2_mosse":
        try:
            return cv2.TrackerMOSSE_create()
        except AttributeError:
            return cv2.legacy.TrackerMOSSE_create()
    raise ValueError(f"Unknown CV2 tracker method: {method!r}")


def _add_detection_field_if_not_exists(dataset: fo.Dataset, field_name: str):
    if str(dataset.media_type) == "video":
        if field_name not in dataset.get_frame_field_schema():
            dataset.add_frame_field(
                field_name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections
            )
    else:
        if field_name not in dataset.get_field_schema():
            dataset.add_sample_field(
                field_name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections
            )
    singleton = fod.Dataset._instances.get(dataset.name)
    if singleton is not None and singleton is not dataset:
        singleton.reload()


def _init_trackers(frame: np.ndarray, detections: fo.Detections, method: str):
    """One tracker per detection. Returns list of (tracker, source_det)."""
    h, w = frame.shape[:2]
    pairs = []
    for det in detections.detections or []:
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(det.bounding_box, w, h)
        tracker = _make_tracker(method)
        ok = tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        if not ok:
            logger.warning("Tracker init failed for detection (label=%s)", det.label)
        pairs.append((tracker, det))
    return pairs


def _update_trackers(frame: np.ndarray, tracker_det_pairs: list) -> fo.Detections:
    h, w = frame.shape[:2]
    new_dets = []
    for tracker, src in tracker_det_pairs:
        try:
            ok, (x, y, tw, th) = tracker.update(frame)
        except Exception as e:
            logger.warning("Tracker update failed: %s", e)
            ok = False

        if ok:
            nx = max(0.0, x / w)
            ny = max(0.0, y / h)
            nw = min(tw / w, 1.0 - nx)
            nh = min(th / h, 1.0 - ny)
            new_bbox = [nx, ny, nw, nh]
        else:
            new_bbox = [0.0, 0.0, 0.0, 0.0]

        new_det = fo.Detection(bounding_box=new_bbox, label=src.label)
        if getattr(src, "index", None) is not None:
            new_det.index = src.index
        new_dets.append(new_det)

    return fo.Detections(detections=new_dets)


def _copy_detections(detections: fo.Detections) -> fo.Detections:
    new_dets = []
    for det in detections.detections or []:
        new_det = fo.Detection(bounding_box=list(det.bounding_box), label=det.label)
        if getattr(det, "index", None) is not None:
            new_det.index = det.index
        new_dets.append(new_det)
    return fo.Detections(detections=new_dets)


def propagate_annotations_cv2(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    method: str = "cv2_csrt",
    sort_field: Optional[str] = None,
    progress: Optional[bool] = True,
) -> dict:
    """
    Propagate bbox annotations using an OpenCV tracker.

    Trackers are initialized once on the first annotated frame and updated
    sequentially — state is maintained across frames, no pairwise re-init.
    """
    media_mode = str(view.media_type)
    if media_mode == "group":
        view = view.flatten()
        media_mode = "image"

    _add_detection_field_if_not_exists(view._dataset, output_annotation_field)

    run_view = (
        view.sort_by(sort_field)
        if (sort_field and view.has_field(sort_field))
        else view
    )

    if media_mode == "image":
        _propagate_image_mode(
            run_view, input_annotation_field, output_annotation_field, method, progress
        )
    elif media_mode == "video":
        _propagate_video_mode(
            run_view, input_annotation_field, output_annotation_field, method, progress
        )
    else:
        raise ValueError(f"Unsupported media type: {media_mode!r}")

    return {}


def _propagate_image_mode(view, input_field, output_field, method, progress):
    samples = list(view.iter_samples(progress=progress))

    seed_idx = None
    for i, s in enumerate(samples):
        ann = s[input_field]
        if ann is not None and getattr(ann, "detections", None):
            seed_idx = i
            break

    if seed_idx is None:
        logger.warning("No annotated seed frame found in '%s' — skipping.", input_field)
        return

    seed = samples[seed_idx]
    seed_img = cv2.imread(seed.filepath)
    tracker_det_pairs = _init_trackers(seed_img, seed[input_field], method)

    seed[output_field] = _copy_detections(seed[input_field])
    seed.save()

    for sample in samples[seed_idx + 1 :]:
        img = cv2.imread(sample.filepath)
        sample[output_field] = _update_trackers(img, tracker_det_pairs)
        sample.save()


def _propagate_video_mode(view, input_field, output_field, method, progress):
    for sample in view.iter_samples(progress=progress):
        frame_numbers = sorted(sample.frames.keys())
        if not frame_numbers:
            continue

        tracker_det_pairs = None
        cap = cv2.VideoCapture(sample.filepath)
        try:
            for fn in frame_numbers:
                # FiftyOne frame numbers are 1-indexed; OpenCV is 0-indexed
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn - 1)
                ok, img = cap.read()
                if not ok:
                    logger.warning("Failed to read frame %d from %s", fn, sample.filepath)
                    continue

                frame_obj = sample.frames[fn]
                ann = frame_obj[input_field]
                has_ann = ann is not None and getattr(ann, "detections", None)

                if tracker_det_pairs is None:
                    if not has_ann:
                        continue
                    tracker_det_pairs = _init_trackers(img, ann, method)
                    frame_obj[output_field] = _copy_detections(ann)
                else:
                    frame_obj[output_field] = _update_trackers(img, tracker_det_pairs)
                frame_obj.save()
        finally:
            cap.release()

        sample.save()
