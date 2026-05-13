import logging
from typing import Union, Optional
import numpy as np
import cv2

import fiftyone as fo
import fiftyone.core.dataset as fod

from .suc_utils import fit_mask_to_bbox, normalized_bbox_to_pixel_coords

logger = logging.getLogger(__name__)

GRABCUT_PROPAGATION_METHODS = ["grabcut"]

# GrabCut hyperparams
_EDGE_KERNEL_SIZE = 3
_EDGE_ITERATIONS = 1
_GRABCUT_ITERATIONS = 5
_CROP_PAD_FACTOR = 0.3   # padding added around bbox before running GrabCut
_GRABCUT_MAX_DIM = 300   # downsample crop to this max dimension before GrabCut


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


def _grabcut_one_detection(target_frame: np.ndarray, det: fo.Detection) -> fo.Detection:
    """Apply GrabCut to propagate a single detection onto target_frame.

    Crops to a padded region around the bbox before running GrabCut so that
    the algorithm operates on a small region rather than the full (e.g. 1080p) frame.
    """
    h, w = target_frame.shape[:2]
    x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(det.bounding_box, w, h)
    bh, bw = y2 - y1, x2 - x1

    # Padded crop bounds (clamped to frame)
    pad_x = int(bw * _CROP_PAD_FACTOR)
    pad_y = int(bh * _CROP_PAD_FACTOR)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop_full = target_frame[cy1:cy2, cx1:cx2]
    full_ch, full_cw = crop_full.shape[:2]

    # Downsample the crop so GrabCut operates on at most _GRABCUT_MAX_DIM px on longest side
    scale = min(1.0, _GRABCUT_MAX_DIM / max(full_ch, full_cw))
    if scale < 1.0:
        cw_ds = max(1, int(round(full_cw * scale)))
        ch_ds = max(1, int(round(full_ch * scale)))
        crop = cv2.resize(crop_full, (cw_ds, ch_ds), interpolation=cv2.INTER_AREA)
    else:
        crop = crop_full
        scale = 1.0
    ch, cw = crop.shape[:2]

    # bbox in downsampled crop-local coordinates
    lx1 = max(0, int(round((x1 - cx1) * scale)))
    ly1 = max(0, int(round((y1 - cy1) * scale)))
    lx2 = min(cw, max(lx1 + 1, int(round((x2 - cx1) * scale))))
    ly2 = min(ch, max(ly1 + 1, int(round((y2 - cy1) * scale))))

    gc_mask = np.zeros((ch, cw), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    source_mask = det.mask

    if source_mask is not None:
        # Resize (not just pad/crop) to the downscaled bbox region
        mh, mw = ly2 - ly1, lx2 - lx1
        mask_resized = cv2.resize(
            (source_mask > 0).astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST
        )
        binary = np.zeros((ch, cw), np.uint8)
        binary[ly1:ly2, lx1:lx2] = mask_resized

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_EDGE_KERNEL_SIZE, _EDGE_KERNEL_SIZE)
        )
        eroded = cv2.erode(binary, kernel, iterations=_EDGE_ITERATIONS)

        # If erosion wiped everything out, use the full binary mask as probable FGD
        fgd_src = eroded if eroded.any() else binary
        gc_mask[fgd_src > 0] = cv2.GC_PR_FGD
        gc_mask[(binary - fgd_src) > 0] = cv2.GC_PR_BGD
        gc_mask[binary == 0] = cv2.GC_PR_BGD

    try:
        if source_mask is not None:
            cv2.grabCut(
                crop, gc_mask, None,
                bgd_model, fgd_model,
                _GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK,
            )
        else:
            rect = (lx1, ly1, lx2 - lx1, ly2 - ly1)
            cv2.grabCut(
                crop, gc_mask, rect,
                bgd_model, fgd_model,
                _GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT,
            )
    except cv2.error as e:
        logger.warning("GrabCut failed for label=%s (%s) — keeping source detection", det.label, e)
        return _copy_one_detection(det)

    refined_ds = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Upsample back to full crop size, then place into full frame
    if scale < 1.0:
        refined_crop = cv2.resize(
            refined_ds, (full_cw, full_ch), interpolation=cv2.INTER_NEAREST
        )
    else:
        refined_crop = refined_ds

    refined = np.zeros((h, w), np.uint8)
    refined[cy1:cy2, cx1:cx2] = refined_crop

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        nx, ny, nw, nh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        new_bbox = [nx / w, ny / h, nw / w, nh / h]
        nx1, ny1, nx2, ny2 = normalized_bbox_to_pixel_coords(new_bbox, w, h)
        new_mask = refined[ny1:ny2, nx1:nx2] if source_mask is not None else None
    else:
        logger.warning("GrabCut found no contour for label=%s — keeping source detection", det.label)
        return _copy_one_detection(det)

    new_det = fo.Detection(bounding_box=new_bbox, label=det.label, mask=new_mask)
    if getattr(det, "index", None) is not None:
        new_det.index = det.index
    return new_det


def _copy_one_detection(det: fo.Detection) -> fo.Detection:
    new_det = fo.Detection(
        bounding_box=list(det.bounding_box), label=det.label, mask=det.mask
    )
    if getattr(det, "index", None) is not None:
        new_det.index = det.index
    return new_det


def _grabcut_frame(target_frame: np.ndarray, source_detections: fo.Detections) -> fo.Detections:
    new_dets = [
        _grabcut_one_detection(target_frame, det)
        for det in (source_detections.detections or [])
    ]
    return fo.Detections(detections=new_dets)


def _copy_detections(detections: fo.Detections) -> fo.Detections:
    return fo.Detections(
        detections=[_copy_one_detection(d) for d in (detections.detections or [])]
    )


def propagate_annotations_grabcut(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
    progress: Optional[bool] = True,
) -> dict:
    """
    Propagate annotations using GrabCut (sequential / tracking paradigm).

    Each frame's output is used as the source for the next frame, so the
    bbox/mask drifts naturally with the object rather than always anchoring
    to the original seed.
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
            run_view, input_annotation_field, output_annotation_field, progress
        )
    elif media_mode == "video":
        _propagate_video_mode(
            run_view, input_annotation_field, output_annotation_field, progress
        )
    else:
        raise ValueError(f"Unsupported media type: {media_mode!r}")

    return {}


def _propagate_image_mode(view, input_field, output_field, progress):
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
    seed[output_field] = _copy_detections(seed[input_field])
    seed.save()

    prev_detections = seed[output_field]

    for sample in samples[seed_idx + 1 :]:
        img = cv2.imread(sample.filepath)
        new_detections = _grabcut_frame(img, prev_detections)
        sample[output_field] = new_detections
        sample.save()
        prev_detections = new_detections


def _propagate_video_mode(view, input_field, output_field, progress):
    for sample in view.iter_samples(progress=progress):
        frame_numbers = sorted(sample.frames.keys())
        if not frame_numbers:
            continue

        prev_detections = None
        cap = cv2.VideoCapture(sample.filepath)
        try:
            for fn in frame_numbers:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn - 1)
                ok, img = cap.read()
                if not ok:
                    logger.warning("Failed to read frame %d from %s", fn, sample.filepath)
                    continue

                frame_obj = sample.frames[fn]
                ann = frame_obj[input_field]
                has_ann = ann is not None and getattr(ann, "detections", None)

                if prev_detections is None:
                    if not has_ann:
                        continue
                    frame_obj[output_field] = _copy_detections(ann)
                    prev_detections = frame_obj[output_field]
                else:
                    new_detections = _grabcut_frame(img, prev_detections)
                    frame_obj[output_field] = new_detections
                    prev_detections = new_detections
                frame_obj.save()
        finally:
            cap.release()

        sample.save()
