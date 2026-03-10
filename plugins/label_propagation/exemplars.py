import logging
from typing import Union, Optional
from collections import defaultdict

import numpy as np
import cv2

import fiftyone as fo

logger = logging.getLogger(__name__)


SUPPORTED_SELECTION_METHODS = [
    "heuristic",
]

SUPPORTED_EXEMPLAR_SELECTION_METHODS = [
    "forward_only",
]


def frame_discontinuity(sample_a, sample_b) -> bool:
    """
    Check if the two samples are "continuous enough".
    Args:
        sample_a: The first sample
        sample_b: The second sample
    Returns:
        bool: True if a large discontinuity is detected between the two samples, False otherwise
    """
    TARGET_SIZE = (256, 256)
    GRAY_CORR_THRESHOLD = 0.9
    HSV_CORR_THRESHOLD = 0.9
    GRAY_DIFF_THRESHOLD = 30

    img_a = cv2.imread(sample_a.filepath)
    img_b = cv2.imread(sample_b.filepath)

    if img_a is None or img_b is None:
        logger.warning(
            f"Failed to read image: {sample_a.filepath if img_a is None else sample_b.filepath}"
        )
        return True

    def get_image_features(img):
        img_resized = cv2.resize(img, TARGET_SIZE)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hsv_hist = cv2.calcHist(
            [hsv], [0, 1], None, [50, 50], [0, 180, 0, 256]
        )
        return gray, hsv, gray_hist, hsv_hist

    gray_a, hsv_a, gray_hist_a, hsv_hist_a = get_image_features(img_a)
    gray_b, hsv_b, gray_hist_b, hsv_hist_b = get_image_features(img_b)

    gray_correlation = cv2.compareHist(
        gray_hist_a, gray_hist_b, cv2.HISTCMP_CORREL
    )
    hsv_correlation = cv2.compareHist(
        hsv_hist_a, hsv_hist_b, cv2.HISTCMP_CORREL
    )
    gray_diff = np.median(cv2.absdiff(gray_a, gray_b))

    is_discontinuous = (
        gray_correlation < GRAY_CORR_THRESHOLD
        or hsv_correlation < HSV_CORR_THRESHOLD
        or gray_diff > GRAY_DIFF_THRESHOLD
    )

    return is_discontinuous


def extract_temporal_segments(
    view: Union[fo.Dataset, fo.DatasetView],
    method: str,
    temporal_segments_field: str,
    sort_field: Optional[str] = None,
) -> None:
    if sort_field and view.has_field(sort_field):
        view = view.sort_by(sort_field)

    if method == "heuristic":
        values = {}
        segment_count = 0
        curr_segment_id = None
        prev_sample = None
        for sample in view:
            if (
                curr_segment_id is None
                or prev_sample is None
                or frame_discontinuity(prev_sample, sample)
            ):
                segment_count += 1
                curr_segment_id = sample.id

            values[sample.id] = fo.Classifications(
                classifications=[
                    fo.Classification(label=str(curr_segment_id), exemplar_score=0.0)
                ]
            )
            prev_sample = sample

        view.set_values(temporal_segments_field, values, key_field="id")
        view.save()
        logger.info(
            f"Extracted {segment_count} temporal segments into '{temporal_segments_field}'"
        )
    else:
        raise NotImplementedError(f"Unsupported method: {method}")


def select_exemplars(
    view: Union[fo.Dataset, fo.DatasetView],
    temporal_segments_field: str,
    method: str,
    sort_field: Optional[str] = None,
) -> None:
    if sort_field and view.has_field(sort_field):
        view = view.sort_by(sort_field)

    if method != "forward_only":
        raise NotImplementedError(f"Unsupported method: {method}")

    segment_to_samples = defaultdict(list)
    for sample in view:
        segs = sample.get_field(temporal_segments_field)
        if segs and segs.classifications:
            for cls in segs.classifications:
                segment_to_samples[cls.label].append(sample)

    updates = {}
    for segment_label, samples in segment_to_samples.items():
        first_id = samples[0].id
        for s in samples:
            segs = s.get_field(temporal_segments_field)
            if not segs or not segs.classifications:
                continue
            new_classes = []
            for cls in segs.classifications:
                if cls.label == segment_label:
                    exemplar_score = 1.0 if s.id == first_id else 0.0
                    new_classes.append(
                        fo.Classification(
                            label=cls.label, exemplar_score=exemplar_score
                        )
                    )
                else:
                    new_classes.append(cls)
            updates[s.id] = fo.Classifications(classifications=new_classes)

    view.set_values(temporal_segments_field, updates, key_field="id")
    view.save()
    logger.info(f"Set exemplar scores in '{temporal_segments_field}'")
