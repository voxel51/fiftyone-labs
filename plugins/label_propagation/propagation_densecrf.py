import logging
from typing import Union, Optional
import numpy as np
import cv2

import fiftyone as fo
import fiftyone.core.dataset as fod

from .suc_utils import normalized_bbox_to_pixel_coords

logger = logging.getLogger(__name__)

DENSECRF_PROPAGATION_METHODS = ["densecrf"]

# DenseCRF hyperparams
_SPATIAL_SMOOTHNESS = 5
_SPATIAL_COMPAT = 5
_BILATERAL_SPATIAL = 40
_BILATERAL_COLOR = 70
_BILATERAL_COMPAT = 50
_UNARY_TEMPERATURE = 5.0
_ITERATIONS = 10
_EPSILON = 1e-8
_CROP_PAD_FACTOR = 0.3
_MAX_DIM = 300


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


def _copy_one_detection(det: fo.Detection) -> fo.Detection:
    new_det = fo.Detection(
        bounding_box=list(det.bounding_box), label=det.label, mask=det.mask
    )
    if getattr(det, "index", None) is not None:
        new_det.index = det.index
    return new_det


def _copy_detections(detections: fo.Detections) -> fo.Detections:
    return fo.Detections(
        detections=[_copy_one_detection(d) for d in (detections.detections or [])]
    )


def _densecrf_one_detection(
    target_frame: np.ndarray, det: fo.Detection
) -> fo.Detection:
    """Apply DenseCRF to refine/propagate a single detection onto target_frame.

    Crops to a padded+downsampled region around the bbox so CRF operates on
    a small region rather than the full (e.g. 1080p) frame.
    """
    try:
        import pydensecrf.densecrf as dcrf
    except ImportError:
        raise ImportError(
            "pydensecrf is required for densecrf propagation. "
            "Install with: pip install pydensecrf"
        )

    h, w = target_frame.shape[:2]
    x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(det.bounding_box, w, h)
    bh, bw = y2 - y1, x2 - x1

    # Padded crop bounds
    pad_x = int(bw * _CROP_PAD_FACTOR)
    pad_y = int(bh * _CROP_PAD_FACTOR)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop_full = target_frame[cy1:cy2, cx1:cx2]
    full_ch, full_cw = crop_full.shape[:2]

    # Downsample crop to at most _MAX_DIM on longest side
    scale = min(1.0, _MAX_DIM / max(full_ch, full_cw))
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

    n_pixels = cw * ch
    source_mask = det.mask

    # Build unary potentials (2 x n_pixels): rows = [bg_prob, fg_prob]
    unary = np.zeros((2, n_pixels), dtype=np.float32)

    if source_mask is not None:
        mh, mw = ly2 - ly1, lx2 - lx1
        mask_resized = cv2.resize(
            (source_mask > 0).astype(np.uint8), (mw, mh),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        mask_in_crop = np.zeros((ch, cw), dtype=np.float32)
        mask_in_crop[ly1:ly2, lx1:lx2] = mask_resized
        mask_flat = mask_in_crop.flatten()

        unary[0] = 1.0 - mask_flat  # background
        unary[1] = mask_flat        # foreground

        if _UNARY_TEMPERATURE != 1.0:
            unary = np.power(np.clip(unary, _EPSILON, 1.0), 1.0 / _UNARY_TEMPERATURE)
    else:
        # Bbox-based soft init: inside bbox = likely FG, outside = likely BG
        for py in range(ch):
            for px in range(cw):
                idx = py * cw + px
                if lx1 <= px < lx2 and ly1 <= py < ly2:
                    unary[0, idx] = 0.3
                    unary[1, idx] = 0.7
                else:
                    unary[0, idx] = 0.9
                    unary[1, idx] = 0.1

    unary = np.clip(unary, _EPSILON, 1.0)
    unary /= unary.sum(axis=0, keepdims=True) + _EPSILON
    unary = -np.log(unary)

    try:
        d = dcrf.DenseCRF2D(cw, ch, 2)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=_SPATIAL_SMOOTHNESS, compat=_SPATIAL_COMPAT)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        d.addPairwiseBilateral(
            sxy=_BILATERAL_SPATIAL,
            srgb=_BILATERAL_COLOR,
            rgbim=rgb,
            compat=_BILATERAL_COMPAT,
        )
        Q = d.inference(_ITERATIONS)
    except Exception as e:
        logger.warning("DenseCRF failed for label=%s (%s) — keeping source", det.label, e)
        return _copy_one_detection(det)

    map_result = np.argmax(Q, axis=0).reshape((ch, cw)).astype(np.uint8)
    refined_ds = (map_result == 1).astype(np.uint8) * 255

    # Upsample back to full crop size, then into full frame
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
        logger.warning("DenseCRF found no contour for label=%s — keeping source", det.label)
        return _copy_one_detection(det)

    new_det = fo.Detection(bounding_box=new_bbox, label=det.label, mask=new_mask)
    if getattr(det, "index", None) is not None:
        new_det.index = det.index
    return new_det


def _densecrf_frame(
    target_frame: np.ndarray, source_detections: fo.Detections
) -> fo.Detections:
    return fo.Detections(
        detections=[
            _densecrf_one_detection(target_frame, det)
            for det in (source_detections.detections or [])
        ]
    )


def propagate_annotations_densecrf(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
    progress: Optional[bool] = True,
) -> dict:
    """
    Propagate annotations using DenseCRF (sequential / tracking paradigm).

    Each frame's output feeds into the next, so the mask refines naturally
    rather than always anchoring to the original seed.
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
        new_detections = _densecrf_frame(img, prev_detections)
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
                    new_detections = _densecrf_frame(img, prev_detections)
                    frame_obj[output_field] = new_detections
                    prev_detections = new_detections
                frame_obj.save()
        finally:
            cap.release()

        sample.save()
