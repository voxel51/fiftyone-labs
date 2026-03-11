import logging
from typing import Union, Optional
from collections import defaultdict

import numpy as np
import cv2

import fiftyone as fo
import fiftyone.core.odm.document as fcod

logger = logging.getLogger(__name__)


SUPPORTED_TEMPORAL_SEGMENTATION_METHODS = [
    "heuristic",
    # TODO(neeraja): add PySceneDetect method [in a follow-up PR]
    # TODO(neeraja): add a fail-safe embedding-based method [in a follow-up PR]
]

SUPPORTED_EXEMPLAR_SCORING_METHODS = [
    "first_frame",
    # TODO(neeraja): add more options post enabling bi-directional propagation
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

    segment_labels = defaultdict(fo.Classifications)

    if method == "heuristic":
        segment_count = 0
        curr_segment_label = None
        prev_sample = None
        for sample in view:
            if (
                prev_sample is None
                or frame_discontinuity(prev_sample, sample)
            ):
                segment_count += 1
                curr_segment_label = fcod.ObjectId()

            segment_labels[sample.id] = fo.Classifications(
                classifications=[
                    fo.Classification(label=str(curr_segment_label), exemplar_score=0.0)
                ]
            )
            prev_sample = sample
    else:
        raise NotImplementedError(f"Unsupported method: {method}")

    view.set_values(temporal_segments_field, segment_labels, key_field="id")
    view.save()
    logger.info(
        f"Extracted {segment_count} temporal segments into '{temporal_segments_field}'"
    )


def select_exemplars(
    view: Union[fo.Dataset, fo.DatasetView],
    temporal_segments_field: str,
    method: str,
    sort_field: Optional[str] = None,
) -> None:
    if method == "first_frame":
        """
        We assume that labels are only propagated forward.
        Hence, the first sample in each segment gets a score of 1.0,
        and the rest get 0.0s.
        """
        segment_ids = set(np.array(
            view.values(f"{temporal_segments_field}.classifications.label")
        ).flatten())
        for seg_id in segment_ids:
            seg_view = view.match(
                {f"{temporal_segments_field}.classifications": {"$elemMatch": {"label": seg_id}}}
            )

            if seg_view.has_field(sort_field):
                seg_view = seg_view.sort_by(sort_field)
            
            first_sample = seg_view.first()
            first_sample_segments = first_sample.get_field(temporal_segments_field).classifications
            for seg in first_sample_segments:
                if seg.label == seg_id:
                    seg.exemplar_score = 1.0
            first_sample[temporal_segments_field].classifications = first_sample_segments
            first_sample.save()
    else:
        raise NotImplementedError(f"Unsupported method: {method}")
    
    view.save()
    logger.info(f"Set exemplar scores in '{temporal_segments_field}'")
