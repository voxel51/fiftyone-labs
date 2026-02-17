import logging
import random
from typing import Union, Optional
import numpy as np
import cv2

import fiftyone as fo

logger = logging.getLogger(__name__)


SUPPORTED_SELECTION_METHODS = [
    "heuristic",
    # TODO(neeraja): add PySceneDetect [in a follow-up PR]
    # TODO(neeraja): add a fail-safe embedding-based method [in a follow-up PR]
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


def extract_exemplar_frames(
    view: Union[fo.Dataset, fo.DatasetView],
    method: str,
    exemplar_frame_field: str,
    sort_field: Optional[str] = None,
) -> None:
    if sort_field and view.has_field(sort_field):
        view = view.sort_by(sort_field)

    if method == "heuristic":
        exemplar_frame_field_values = {}
        exemplar_count = 0
        curr_exemplar_id = view.first().id
        prev_sample = view[curr_exemplar_id]
        for ii, sample in enumerate(view):
            if (sample.id == curr_exemplar_id) or frame_discontinuity(prev_sample, sample):
                is_exemplar = True
                exemplar_count += 1
                curr_exemplar_id = sample.id
            else:
                is_exemplar = False
            exemplar_frame_field_values[sample.id] = fo.DynamicEmbeddedDocument(
                is_exemplar=is_exemplar,
                exemplar_assignment=[curr_exemplar_id]
                if not is_exemplar
                else [],
            )
            prev_sample = sample
        
        view.set_values(exemplar_frame_field, exemplar_frame_field_values, key_field="id")
        view.save()
        logger.info(f"Extracted {exemplar_count} exemplar frames and stored in field '{exemplar_frame_field}'")  # type: ignore[arg-type]

    else:
        raise NotImplementedError(f"Unsupported method: {method}")

    return
