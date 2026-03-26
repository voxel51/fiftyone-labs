"""Unit tests for functions in the label_propagation plugin."""

from __future__ import annotations

import pytest
from pathlib import Path
import sys
from typing import Tuple
import numpy as np
import cv2

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz
from fiftyone.core.expressions import ViewField as F

_TEST_PKG_DIR = Path(__file__).resolve().parent.parent
PLUGINS_DIR = _TEST_PKG_DIR.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from label_propagation.suc_utils import evaluate_matched  # type: ignore

import logging
logger = logging.getLogger(__name__)

PROPAGATION_SUC_THRESHOLD = 0.9


def frame_discontinuity_test(
    img_a: np.ndarray,
    img_b: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[float, float, float]:
    """
    Check if the two image arrays are "continuous enough".

    Args:
        img_a: First image as BGR numpy array (e.g., from cv2.imread)
        img_b: Second image as BGR numpy array

    Returns:
        gray_correlation: float
        hsv_correlation: float
        gray_diff: float
    """
    if img_a is None or img_b is None:
        return 0.0, 0.0, 0.0

    def get_image_features(img):
        img_resized = cv2.resize(img, target_size)
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

    return gray_correlation, hsv_correlation, gray_diff


@pytest.fixture()
def image_dataset(request):
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )

    if "labels_test" in dataset._dataset.get_field_schema():
        dataset._dataset.delete_sample_field(
            "labels_test", error_level=2
        )
    dataset._dataset.add_sample_field(
        "labels_test",
        fo.EmbeddedDocumentField,
        embedded_doc_type=fo.Detections,
    )

    return dataset


def test_frame_discontinuity(image_dataset):

    GREY_CORRELATION_THRESHOLDS = []
    HSV_CORRELATION_THRESHOLDS = []
    GRAY_DIFF_THRESHOLDS = []

    sequences = image_dataset.distinct("tags")
    sequences.remove("val")
    for sequence in sequences:
        sequence_view = image_dataset.match_tags([sequence]).sort_by("frame_number")
        sequence_view.set_values("labels_test", [fo.Detections(detections=[]),]*len(sequence_view))
        first_sample = sequence_view.first()
        first_sample["labels_test"] = first_sample["ground_truth"]
        first_sample.save()

        sequence_ids = sequence_view.values("id")
        first_sample_id = sequence_ids[0]
        first_image = cv2.cvtColor(cv2.imread(sequence_view[first_sample_id].filepath), cv2.COLOR_BGR2RGB)

        propagation_scores = []
        discontinuity_scores = []

        for sample_id in sequence_ids[1:]:

            pair_view = sequence_view.select([first_sample_id, sample_id])
            ctx = {
                "dataset": pair_view._dataset,
                "view": pair_view,
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

            pred_detections = pair_view.values("labels_test_propagated")
            gt_detections = pair_view.values("ground_truth")

            score = evaluate_matched(pred_detections[-1], gt_detections[-1])
            propagation_scores.append(score)

            sample_image = cv2.cvtColor(cv2.imread(sequence_view[sample_id].filepath), cv2.COLOR_BGR2RGB)
            gray_correlation, hsv_correlation, gray_diff = frame_discontinuity_test(
                first_image, sample_image
            )
            discontinuity_scores.append((gray_correlation, hsv_correlation, gray_diff))

            if score < PROPAGATION_SUC_THRESHOLD:
                break
    
        print("\n finished sequence: ", sequence, "Suggested thresholds:")
        print(f"gray_correlation >= {gray_correlation}")
        print(f"hsv_correlation >= {hsv_correlation}")
        print(f"gray_diff <= {gray_diff}")

        GREY_CORRELATION_THRESHOLDS.append(gray_correlation)
        HSV_CORRELATION_THRESHOLDS.append(hsv_correlation)
        GRAY_DIFF_THRESHOLDS.append(gray_diff)

        logger.info(f"\n\n Cumulative Statistics:")
        logger.info(f"gray_correlation: min = {np.min(GREY_CORRELATION_THRESHOLDS)}, max = {np.max(GREY_CORRELATION_THRESHOLDS)}, median = {np.median(GREY_CORRELATION_THRESHOLDS)}")
        logger.info(f"hsv_correlation: min = {np.min(HSV_CORRELATION_THRESHOLDS)}, max = {np.max(HSV_CORRELATION_THRESHOLDS)}, median = {np.median(HSV_CORRELATION_THRESHOLDS)}")
        logger.info(f"gray_diff: min = {np.min(GRAY_DIFF_THRESHOLDS)}, max = {np.max(GRAY_DIFF_THRESHOLDS)}, median = {np.median(GRAY_DIFF_THRESHOLDS)}")