import os
import sys
import importlib.util
from typing import Tuple, Union, Optional
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

import fiftyone.core.labels as fol


detection_area = lambda det: det["bounding_box"][2] * det["bounding_box"][3]


def drop_zero_area_detections(detections_container: fol.Detections):
    """
    Drop zero-area boxes before set-based matching.

    Those entries are treated as non-participating placeholders so they neither
    inflate scores (success-rate padding) nor shrink ``evaluate`` via
    ``max(G, P)`` without contributing IoU.
    """
    if (
        not hasattr(detections_container, "detections")
        or detections_container.detections is None
    ):
        return fol.Detections(detections=[])
    return fol.Detections(
        detections=[
            d
            for d in detections_container.detections  # type: ignore[attr-defined]
            if detection_area(d) > 0
        ]
    )


def load_local_utils(filename: str, module_name: str):
    """
    Load this plugin's local `utils.py` under a unique module name.

    This avoids collisions with other plugins that may also define a
    top-level `utils` module (for example, `@51labs/zero-shot-coreset-selection`).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(here, filename)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load module {module_name} from {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def fit_mask_to_bbox(
    mask: np.ndarray, bbox_size: Tuple[int, int]
) -> np.ndarray:
    """
    Pads or crops the mask to the bounding box size.
    Args:
        mask: np.ndarray of shape (mask_height, mask_width)
        bbox_size: Tuple[int, int] of the bounding box size (height, width)
    Returns:
        np.ndarray of shape (height, width)
    """
    return np.pad(
        mask,
        [
            (0, max(0, bbox_size[0] - mask.shape[0])),
            (0, max(0, bbox_size[1] - mask.shape[1])),
        ],
    )[: bbox_size[0], : bbox_size[1]]


def normalized_bbox_to_pixel_coords(bbox, image_width, image_height):
    """
    Convert normalized bounding box [x, y, width, height] to pixel coordinates.

    Args:
        bbox: Normalized bounding box [x, y, width, height]
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        tuple: (x1, y1, x2, y2) pixel coordinates
    """
    x1 = int(bbox[0] * image_width)
    y1 = int(bbox[1] * image_height)
    x2 = int((bbox[0] + bbox[2]) * image_width)
    y2 = int((bbox[1] + bbox[3]) * image_height)
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2


def box_iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a["bounding_box"]
    bx, by, bw, bh = box_b["bounding_box"]

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area

    if union == 0:
        return 1.0

    return inter_area / union


def evaluate(original_detections, propagated_detections):
    """
    Evaluate the propagation against the original detection.
    Args:
        original_detections: The original detections
        propagated_detections: The propagated detections
    Returns:
        float: The evaluation score
    """
    # TODO(neeraja): replace this with a more standard evaluation metric

    # for now, only evaluates bounding boxes
    # TODO(neeraja): implement for masks

    if not hasattr(original_detections, "detections") and not hasattr(
        propagated_detections, "detections"
    ):
        return 1.0
    elif not hasattr(original_detections, "detections") or not hasattr(
        propagated_detections, "detections"
    ):
        return 0.0

    original_detections = drop_zero_area_detections(original_detections)
    propagated_detections = drop_zero_area_detections(propagated_detections)

    G = len(original_detections.detections)
    P = len(propagated_detections.detections)

    if max(G, P) == 0:
        return 1.0
    elif min(G, P) == 0:
        return 0.0

    # IoU matrix: shape (G, P)
    iou_matrix = np.zeros((G, P), dtype=np.float32)
    for i, gt in enumerate(original_detections.detections):
        for j, pred in enumerate(propagated_detections.detections):
            iou_matrix[i, j] = box_iou(gt, pred)

    # Hungarian finds MIN cost → negate IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    total_iou = 0.0
    for i, j in zip(row_ind, col_ind):
        total_iou += iou_matrix[i, j]

    # Unmatched predictions/ground_truths contribute IoU = 0 implicitly
    return total_iou / max(G, P)


def evaluate_matched(original_detections, propagated_detections):
    """
    Evaluate the propagation against the original detection.
    Args:
        original_detections: The original detections
        propagated_detections: The propagated detections
    Returns:
        float: The evaluation score
    """
    # TODO(neeraja): implement for masks
    if len(original_detections.detections) == 0:
        return 0.0

    assert len(original_detections.detections) == len(
        propagated_detections.detections
    )
    total_iou = 0.0
    n = 0
    for od, pd in zip(
        original_detections.detections, propagated_detections.detections
    ):
        if detection_area(od) <= 0 and detection_area(pd) <= 0:
            continue
        total_iou += box_iou(od, pd)
        n += 1
    if n == 0:
        return 1.0
    return total_iou / n


def sort_detections_by_index(detections):
    """``fol.Detections`` with ``detections`` sorted by ``index`` (indices must be set)."""
    if detections is None or not detections.detections:
        return fol.Detections(detections=[])
    return fol.Detections(
        detections=sorted(
            detections.detections,  # type: ignore[attr-defined]
            key=lambda d: d.index,
        )
    )


def _coerce_detections_field(value):
    if value is None:
        return fol.Detections(detections=[])
    return value


def _iter_gt_pred_pairs(gt_list, pred_list):
    """Yield ``(gt, pred)`` per sample (image) or per frame (video: ``frames.*`` values)."""
    video = bool(gt_list) and isinstance(gt_list[0], (list, tuple))
    pred_video = bool(pred_list) and isinstance(pred_list[0], (list, tuple))
    if video != pred_video:
        raise ValueError("gt and pred disagree on image vs frame-wise layout")
    if video:
        for sg, sp in zip(gt_list, pred_list):
            if len(sg) != len(sp):
                raise ValueError("Mismatched frame count between gt and pred")
            yield from zip(sg, sp)
    else:
        yield from zip(gt_list, pred_list)


def _all_indices_nonnull(gt_list, pred_list):
    for gt, pred in _iter_gt_pred_pairs(gt_list, pred_list):
        gt, pred = _coerce_detections_field(gt), _coerce_detections_field(pred)
        for d in (*gt.detections, *pred.detections):  # type: ignore[attr-defined]
            if getattr(d, "index", None) is None:
                return False
    return True


def _score_gt_pred_pair(gt, pred, use_index):
    gt, pred = _coerce_detections_field(gt), _coerce_detections_field(pred)
    if use_index:
        ps = sort_detections_by_index(pred)
        gs = sort_detections_by_index(gt)
        gn, pn = len(gs.detections), len(ps.detections)  # type: ignore[attr-defined]
        if gn > 0 and pn > 0 and gn == pn:
            return evaluate_matched(ps, gs)
    return evaluate(gt, pred)


def evaluate_detections(view, pred_field: str, gt_field: str):
    """
    Per-sample (image) or flattened per-frame (``frames.*`` video) scores vs
    ``fo.utils.eval.detection.evaluate_detections`` field pattern, no ``iou``.

    If every detection has non-``None`` ``index``, uses :func:`evaluate_matched` on
    index-sorted preds/gts when both sides have the same positive count; otherwise
    :func:`evaluate`. Labels ignored.
    """
    gt_list = list(view.values(gt_field))
    pred_list = list(view.values(pred_field))
    if len(gt_list) != len(pred_list):
        raise ValueError(f"{len(gt_list)=} != {len(pred_list)=}")
    use_index = _all_indices_nonnull(gt_list, pred_list)
    return [
        _score_gt_pred_pair(gt, pred, use_index)
        for gt, pred in _iter_gt_pred_pairs(gt_list, pred_list)
    ]


def evaluate_success_rate(original_detections, propagated_detections):
    """
    The success plot represents the percentage of frames for which the IoU exceeds a threshold,
    with respect to different thresholds.
    The area under the success plot is taken as an overall success measure.
    Args:
        original_detections: The original detections
        propagated_detections: The propagated detections
    Returns:
        float: The evaluation score
    """
    if not hasattr(original_detections, "detections") and not hasattr(
        propagated_detections, "detections"
    ):
        return 1.0
    elif not hasattr(original_detections, "detections"):
        return float(len(propagated_detections.detections) == 0)
    elif not hasattr(propagated_detections, "detections"):
        return float(len(original_detections.detections) == 0)

    # TODO(neeraja): implement for masks
    original_detections = drop_zero_area_detections(original_detections)
    propagated_detections = drop_zero_area_detections(propagated_detections)

    G = len(original_detections.detections)
    P = len(propagated_detections.detections)

    if max(G, P) == 0:
        return 1.0
    elif min(G, P) == 0:
        return 0.0

    # IoU matrix: shape (G, P)
    iou_matrix = np.zeros((G, P), dtype=np.float32)
    for i, gt in enumerate(original_detections.detections):
        for j, pred in enumerate(propagated_detections.detections):
            iou_matrix[i, j] = box_iou(gt, pred)

    # Hungarian finds MIN cost → negate IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Get unmatched detections (all positive area here)
    unmatched_original_ind = [ii for ii in range(G) if ii not in row_ind]
    unmatched_propagated_ind = [jj for jj in range(P) if jj not in col_ind]

    # Get matched IoUs
    ious: list[float] = [
        float(iou_matrix[i, j]) for i, j in zip(row_ind, col_ind)
    ]
    ious.extend(
        [0.0] * (len(unmatched_original_ind) + len(unmatched_propagated_ind))
    )
    ious = sorted(ious, reverse=True)

    # Compute area under success curve (threshold vs success rate)
    area_under_curve = 0
    count_thresh = 0  # the index of ious containing iou < iou_thresh
    for ii, iou_thresh in enumerate(sorted(np.unique(ious), reverse=True)):
        while count_thresh < len(ious) and ious[count_thresh] >= iou_thresh:
            count_thresh += 1
        area_under_curve += iou_thresh * (count_thresh - ii) / len(ious)

    return area_under_curve


def evaluate_success_rate_matched(original_detections, propagated_detections):
    """
    The success plot represents the percentage of frames for which the IoU exceeds a threshold,
    with respect to different thresholds.
    The area under the success plot is taken as an overall success measure.
    """
    # TODO(neeraja): implement for masks
    if (
        len(original_detections.detections) == 0
        or len(propagated_detections.detections) == 0
    ):
        return 0.0

    assert len(original_detections.detections) == len(
        propagated_detections.detections
    )

    ious = []
    for od, pd in zip(
        original_detections.detections, propagated_detections.detections
    ):
        if detection_area(od) <= 0 and detection_area(pd) <= 0:
            continue
        ious.append(box_iou(od, pd))
    if not ious:
        return 1.0
    ious = sorted(ious)[::-1]

    # Compute area under success curve (threshold vs success rate)
    area_under_curve = 0
    count_thresh = 0  # the index of ious containing iou < iou_thresh
    for ii, iou_thresh in enumerate(sorted(np.unique(ious), reverse=True)):
        while count_thresh < len(ious) and ious[count_thresh] >= iou_thresh:
            count_thresh += 1
        area_under_curve += iou_thresh * (count_thresh - ii) / len(ious)
    return area_under_curve
