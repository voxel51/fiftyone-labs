import logging
import random
from typing import Union
import numpy as np
import cv2

import fiftyone as fo

logger = logging.getLogger(__name__)


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
    CORR_THRESHOLD = 0.7
    MSE_THRESHOLD = 1000

    img_a = cv2.imread(sample_a.filepath)
    img_b = cv2.imread(sample_b.filepath)

    def get_image_features(img):
        img_resized = cv2.resize(img, TARGET_SIZE)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return gray, hist

    gray_a, hist_a = get_image_features(img_a)
    gray_b, hist_b = get_image_features(img_b)

    # Compare histograms using correlation (returns value between 0 and 1, where 1 is identical)
    correlation = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    # Also compute mean squared error for additional check
    mse = np.mean((gray_a.astype(float) - gray_b.astype(float)) ** 2)

    is_discontinuous = correlation < CORR_THRESHOLD or mse > MSE_THRESHOLD

    return is_discontinuous


def extract_exemplar_frames(
    view: Union[fo.Dataset, fo.DatasetView],
    method: str,
    exemplar_frame_field: str,
    sort_field: str,
) -> None:
    if method == "uniform":
        PERIOD = 2
        # every PERIOD-th sample is an exemplar
        # first frame is an exemplar
        curr_exemplar_id = view.sort_by(sort_field).first().id
        for ii, sample in enumerate(view.sort_by(sort_field)):
            if ii % PERIOD == 0:
                curr_exemplar_id = sample.id
                is_exemplar = True
            else:
                is_exemplar = False
            sample[exemplar_frame_field] = fo.DynamicEmbeddedDocument(
                is_exemplar=is_exemplar,
                exemplar_assignment=[curr_exemplar_id]
                if not is_exemplar
                else [],
            )
            sample.save()
        logger.info(f"Extracted {len(view) // PERIOD} exemplar frames and stored in field '{exemplar_frame_field}'")  # type: ignore[arg-type]

    elif method == "heuristic":
        exemplar_count = 0
        curr_exemplar_id = view.sort_by(sort_field).first().id
        prev_sample = view[curr_exemplar_id]
        for ii, sample in enumerate(view.sort_by(sort_field)):
            if sample.id == curr_exemplar_id:
                is_exemplar = True
                exemplar_count += 1
            elif frame_discontinuity(prev_sample, sample):
                is_exemplar = True
                exemplar_count += 1
                curr_exemplar_id = sample.id
            else:
                is_exemplar = False
            sample[exemplar_frame_field] = fo.DynamicEmbeddedDocument(
                is_exemplar=is_exemplar,
                exemplar_assignment=[curr_exemplar_id]
                if not is_exemplar
                else [],
            )
            sample.save()
            prev_sample = sample
        logger.info(f"Extracted {exemplar_count} exemplar frames and stored in field '{exemplar_frame_field}'")  # type: ignore[arg-type]

    else:
        raise NotImplementedError(f"Unsupported method: {method}")

    return
