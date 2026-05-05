import pytest
import numpy as np
import sys
from pathlib import Path

import fiftyone.zoo as foz

_TEST_PKG_DIR = Path(__file__).resolve().parent
PLUGINS_DIR = _TEST_PKG_DIR.parent.parent / "plugins"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from label_propagation.suc_utils import (
    fit_mask_to_bbox,
    normalized_bbox_to_pixel_coords,
    box_iou,
    evaluate,
    evaluate_matched,
    evaluate_success_rate,
    evaluate_success_rate_matched,
)


class TestBasicUtils:
    def test_fit_mask_to_bbox(self):
        # Test case: mask smaller than bbox_size (should pad)
        mask_small = np.array([[1, 0], [0, 0]])
        bbox_size = (5, 5)
        result = fit_mask_to_bbox(mask_small, bbox_size)
        assert result.shape == (5, 5)
        assert np.array_equal(result[:2, :2], mask_small)
        assert np.all(result[2:, :] == 0)
        assert np.all(result[:, 2:] == 0)

        # Test case: mask larger than bbox_size (should crop)
        mask_large = np.ones((10, 10))
        bbox_size = (5, 5)
        result = fit_mask_to_bbox(mask_large, bbox_size)
        assert result.shape == (5, 5)
        assert np.all(result == 1)

        # Test case: smaller in 1 dim, larger in another dim
        mask_mixed = np.ones((2, 10))
        bbox_size = (5, 3)
        result = fit_mask_to_bbox(mask_mixed, bbox_size)
        assert result.shape == (5, 3)
        assert np.all(result[:2, :3] == 1)  # Original mask portion
        assert np.all(result[2:, :] == 0)  # Padded portion

    def test_normalized_bbox_to_pixel_coords(self):
        # Test case: normal case
        bbox = [0.1, 0.2, 0.3, 0.4]
        image_width, image_height = 200, 100
        x1, y1, ww, hh = normalized_bbox_to_pixel_coords(
            bbox, image_width, image_height
        )
        assert x1 == 20
        assert y1 == 20
        assert ww == 80
        assert hh == 60

        # Test case: box spilling out of the image
        bbox = [0.9, 0.8, 0.3, 0.3]
        image_width, image_height = 200, 100
        x1, y1, ww, hh = normalized_bbox_to_pixel_coords(
            bbox, image_width, image_height
        )
        assert x1 == 180
        assert y1 == 80
        assert ww == 200
        assert hh == 100

    def test_box_iou(self):
        # Test case: overlapping boxes
        box_a = {"bounding_box": [0.1, 0.1, 0.2, 0.2]}
        box_b = {"bounding_box": [0.2, 0.2, 0.2, 0.2]}
        iou = box_iou(box_a, box_b)
        assert 0 < iou < 1
        assert abs(iou - 1 / 7) < 1e-6

        # Test case: non-overlapping boxes
        box_a = {"bounding_box": [0.1, 0.1, 0.2, 0.2]}
        box_b = {"bounding_box": [0.3, 0.3, 0.2, 0.2]}
        iou = box_iou(box_a, box_b)
        assert iou < 1e-6

        # Test case: identical boxes
        box_a = {"bounding_box": [0.1, 0.3, 0.2, 0.2]}
        box_b = {"bounding_box": [0.1, 0.3, 0.2, 0.2]}
        iou = box_iou(box_a, box_b)
        assert abs(iou - 1) < 1e-6


class MockDetections:
    """Mock object to simulate detections with .detections attribute"""

    def __init__(self, detections):
        if detections is not None:
            self.detections = detections


class TestEvalMetrics:
    def test_evaluate(self):
        # Test case: one of the lists empty
        original = MockDetections([])
        propagated = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        score = evaluate(original, propagated)
        assert score == 0.0
        # Also test the other way around
        original = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        propagated = MockDetections([])
        score = evaluate(original, propagated)
        assert score == 0.0

        # Test case: both lists have 1 object each (overlapping)
        original = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        propagated = MockDetections([{"bounding_box": [0.2, 0.2, 0.2, 0.2]}])
        score = evaluate(original, propagated)
        assert abs(score - 1 / 7) < 1e-6

        # Test case: original_detections has 1 object, propagated has 2 (one overlapping mostly)
        original = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        propagated = MockDetections(
            [
                {"bounding_box": [0.35, 0.35, 0.2, 0.2]},
                {"bounding_box": [0.11, 0.11, 0.2, 0.2]},
            ]
        )
        score = evaluate(original, propagated)
        assert 0 < score < 0.5  # ~0.4112

    def test_evaluate_matched(self):
        # Test case: both lists have 2 object each, one mostly overlapping
        original = MockDetections(
            [
                {"bounding_box": [0.1, 0.1, 0.2, 0.2]},
                {"bounding_box": [0.5, 0.6, 0.2, 0.2]},
            ]
        )
        propagated = MockDetections(
            [
                {"bounding_box": [0.11, 0.11, 0.2, 0.2]},
                {"bounding_box": [0.8, 0.6, 0.2, 0.2]},
            ]
        )
        score = evaluate_matched(original, propagated)
        assert 0 < score < 0.5

        # Test case: swapped order
        original = MockDetections(
            [
                {"bounding_box": [0.1, 0.1, 0.2, 0.2]},
                {"bounding_box": [0.5, 0.6, 0.2, 0.2]},
            ]
        )
        propagated = MockDetections(
            [
                {"bounding_box": [0.8, 0.6, 0.2, 0.2]},
                {"bounding_box": [0.11, 0.11, 0.2, 0.2]},
            ]
        )
        score = evaluate_matched(original, propagated)
        assert score < 1e-6

    def test_evaluate_success_rate_parity(self):
        # Test case: both lists have 1 object each (overlapping)
        original = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        propagated = MockDetections([{"bounding_box": [0.2, 0.2, 0.2, 0.2]}])
        score = evaluate(original, propagated)
        score_suc = evaluate_success_rate(original, propagated)
        assert abs(score - score_suc) < 1e-6

        # Test case: original_detections has 1 object, propagated has 2 (one overlapping mostly)
        original = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        propagated = MockDetections(
            [
                {"bounding_box": [0.35, 0.35, 0.2, 0.2]},
                {"bounding_box": [0.11, 0.11, 0.2, 0.2]},
            ]
        )
        score = evaluate(original, propagated)
        score_suc = evaluate_success_rate(original, propagated)
        assert abs(score - score_suc) < 1e-6

        # Test case from an example: exactly matching boxes
        original = MockDetections(
            [
                {"bounding_box": [0.869071, 0.448725, 0.108700, 0.243545]},
                {"bounding_box": [0.836372, 0.592978, 0.027977, 0.050664]},
            ]
        )
        propagated = MockDetections(
            [
                {"bounding_box": [0.836372, 0.592978, 0.027977, 0.050664]},
                {"bounding_box": [0.869071, 0.448725, 0.108700, 0.243545]},
            ]
        )
        score = evaluate(original, propagated)
        score_suc = evaluate_success_rate(original, propagated)
        assert abs(score - score_suc) < 1e-6

        # Test case from an example: almost match
        original = MockDetections(
            [
                {"bounding_box": [0.3265, 0.2241, 0.1307, 0.2330]},
                {"bounding_box": [0.3663, 0.3451, 0.4114, 0.6512]},
            ]
        )
        propagated = MockDetections(
            [
                {"bounding_box": [0.326625, 0.225, 0.13125, 0.2306]},
                {
                    "bounding_box": [
                        0.3704375,
                        0.3473333333333333,
                        0.4079375,
                        0.6528,
                    ]
                },
            ]
        )
        score = evaluate(original, propagated)
        score_suc = evaluate_success_rate(original, propagated)
        assert abs(score - score_suc) < 1e-6

        # Zero-area placeholders are dropped before matching, so ``evaluate`` and
        # ``evaluate_success_rate`` stay aligned (no artificial IoU=1 padding).
        original = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        propagated = MockDetections(
            [
                {"bounding_box": [0.1, 0.1, 0.2, 0.2]},
                {"bounding_box": [0.0, 0.0, 0.0, 0.0]},
            ]
        )
        score = evaluate(original, propagated)
        score_suc = evaluate_success_rate(original, propagated)
        assert abs(score - 1.0) < 1e-6
        assert abs(score_suc - 1.0) < 1e-6
        assert abs(score - score_suc) < 1e-6

        score_rev = evaluate(propagated, original)
        score_suc_rev = evaluate_success_rate(propagated, original)
        assert abs(score_rev - 1.0) < 1e-6
        assert abs(score_suc_rev - 1.0) < 1e-6
        assert abs(score_rev - score_suc_rev) < 1e-6

    def test_evaluate_success_rate_nulls(self):
        # Test case: null detections
        original = MockDetections(None)
        propagated = MockDetections(None)
        score = evaluate_success_rate(original, propagated)
        assert abs(score - 1) < 1e-6

        # Test case: empty detections
        original = MockDetections(None)
        propagated = MockDetections([])
        score = evaluate_success_rate(original, propagated)
        assert abs(score - 1) < 1e-6

        # Test case: null bounding boxes
        # Note: we should never have null bounding boxes in the first place
        original = MockDetections(
            [
                {"bounding_box": [0.0, 0.0, 0.0, 0.0]},
            ]
        )
        propagated = MockDetections(
            [
                {"bounding_box": [0.0, 0.0, 0.0, 0.0]},
            ]
        )
        score = evaluate_success_rate(original, propagated)
        assert abs(score - 1) < 1e-6

        # Test case: null bounding boxes vs empty detections
        # Note: we should never have null bounding boxes in the first place
        original = MockDetections([])
        propagated = MockDetections(
            [
                {"bounding_box": [0.0, 0.0, 0.0, 0.0]},
                {"bounding_box": [0.0, 0.0, 0.0, 0.0]},
            ]
        )
        score = evaluate_success_rate(original, propagated)
        assert abs(score - 1) < 1e-6

        # Test case: one of the lists has extra zero-size boxes
        original = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        propagated = MockDetections(
            [
                {"bounding_box": [0.1, 0.1, 0.2, 0.2]},
                {"bounding_box": [0.0, 0.0, 0.0, 0.0]},
            ]
        )
        score = evaluate_success_rate(original, propagated)
        assert abs(score - 1) < 1e-6
        score = evaluate_success_rate(propagated, original)
        assert abs(score - 1) < 1e-6

    def test_evaluate_success_rate(self):
        # Test case: one of the lists empty
        original = MockDetections([])
        propagated = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        score = evaluate_success_rate(original, propagated)
        assert score == 0.0
        score = evaluate_success_rate(propagated, original)
        assert score == 0.0

        # Test case: both lists have 1 object each (overlapping)
        original = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        propagated = MockDetections([{"bounding_box": [0.2, 0.2, 0.2, 0.2]}])
        score = evaluate_success_rate(original, propagated)
        assert abs(score - 1 / 7) < 1e-6
        score = evaluate_success_rate(propagated, original)
        assert abs(score - 1 / 7) < 1e-6

        # Test case: original_detections has 1 object, propagated has 2 (one overlapping mostly)
        original = MockDetections([{"bounding_box": [0.1, 0.1, 0.2, 0.2]}])
        propagated = MockDetections(
            [
                {"bounding_box": [0.35, 0.35, 0.2, 0.2]},
                {"bounding_box": [0.11, 0.11, 0.2, 0.2]},
            ]
        )
        score = evaluate_success_rate(original, propagated)
        assert 0 < score < 0.5

        # Test case from an example: exactly matching boxes
        original = MockDetections(
            [
                {"bounding_box": [0.869071, 0.448725, 0.108700, 0.243545]},
                {"bounding_box": [0.836372, 0.592978, 0.027977, 0.050664]},
            ]
        )
        propagated = MockDetections(
            [
                {"bounding_box": [0.836372, 0.592978, 0.027977, 0.050664]},
                {"bounding_box": [0.869071, 0.448725, 0.108700, 0.243545]},
            ]
        )
        score = evaluate_success_rate(original, propagated)
        assert abs(score - 1) < 1e-6

        # Test case from an example: almost match
        original = MockDetections(
            [
                {"bounding_box": [0.3265, 0.2241, 0.1307, 0.2330]},
                {"bounding_box": [0.3663, 0.3451, 0.4114, 0.6512]},
            ]
        )
        propagated = MockDetections(
            [
                {"bounding_box": [0.326625, 0.225, 0.13125, 0.2306]},
                {
                    "bounding_box": [
                        0.3704375,
                        0.3473333333333333,
                        0.4079375,
                        0.6528,
                    ]
                },
            ]
        )
        score = evaluate_success_rate(original, propagated)
        assert abs(score - 1) < 1e-1

    def test_evaluate_success_rate_matched(self):
        # Test case: both lists have 2 object each, one mostly overlapping
        original = MockDetections(
            [
                {"bounding_box": [0.1, 0.1, 0.2, 0.2]},
                {"bounding_box": [0.5, 0.6, 0.2, 0.2]},
            ]
        )
        propagated = MockDetections(
            [
                {"bounding_box": [0.11, 0.11, 0.2, 0.2]},
                {"bounding_box": [0.8, 0.6, 0.2, 0.2]},
            ]
        )
        score = evaluate_success_rate_matched(original, propagated)
        assert 0 < score < 0.5

        # Test case: swapped order
        original = MockDetections(
            [
                {"bounding_box": [0.1, 0.1, 0.2, 0.2]},
                {"bounding_box": [0.5, 0.6, 0.2, 0.2]},
            ]
        )
        propagated = MockDetections(
            [
                {"bounding_box": [0.8, 0.6, 0.2, 0.2]},
                {"bounding_box": [0.11, 0.11, 0.2, 0.2]},
            ]
        )
        score = evaluate_success_rate_matched(original, propagated)
        assert score < 1e-6

    def test_coco_eval(self, tmp_path):
        """
        Illustrate FiftyOne COCO-style `evaluate_detections` vs `evaluate()`.

        COCO-style metrics use a fixed IoU threshold (TP/FP/FN), class matching,
        and global precision/recall; `evaluate()` is Hungarian mean IoU over
        max(G, P). Scores are printed only (no assertions).
        """
        pytest.importorskip("PIL", reason="PIL required for sample images")
        import fiftyone as fo
        from fiftyone import Detection, Detections
        from PIL import Image

        img_path = tmp_path / "coco_eval.jpg"
        Image.new("RGB", (100, 100), color=(0, 0, 0)).save(img_path)

        cases = [
            (
                "perfect_match",
                [Detection(label="obj", bounding_box=[0.1, 0.1, 0.2, 0.2])],
                [Detection(label="obj", bounding_box=[0.1, 0.1, 0.2, 0.2])],
            ),
            (
                "partial_overlap_iou_1_7",
                [Detection(label="obj", bounding_box=[0.1, 0.1, 0.2, 0.2])],
                [Detection(label="obj", bounding_box=[0.2, 0.2, 0.2, 0.2])],
            ),
            (
                "one_gt_two_preds_hungarian",
                [Detection(label="obj", bounding_box=[0.1, 0.1, 0.2, 0.2])],
                [
                    Detection(
                        label="obj", bounding_box=[0.35, 0.35, 0.2, 0.2]
                    ),
                    Detection(
                        label="obj", bounding_box=[0.11, 0.11, 0.2, 0.2]
                    ),
                ],
            ),
            (
                "two_gt_one_pred",
                [
                    Detection(label="obj", bounding_box=[0.1, 0.1, 0.2, 0.2]),
                    Detection(label="obj", bounding_box=[0.5, 0.6, 0.2, 0.2]),
                ],
                [Detection(label="obj", bounding_box=[0.11, 0.11, 0.2, 0.2])],
            ),
            (
                "label_mismatch_same_box",
                [Detection(label="gt_cls", bounding_box=[0.1, 0.1, 0.2, 0.2])],
                [
                    Detection(
                        label="pred_cls", bounding_box=[0.1, 0.1, 0.2, 0.2]
                    )
                ],
            ),
        ]

        for iou_thresh in (0.5, 0.9):
            print(f"\n--- COCO eval iou={iou_thresh} ---")
            for name, gt_list, pr_list in cases:
                gt_fo = Detections(detections=gt_list)
                pr_fo = Detections(detections=pr_list)
                eval_score = evaluate(gt_fo, pr_fo)

                ds = fo.Dataset()
                try:
                    ds.add_sample(
                        fo.Sample(
                            filepath=str(img_path),
                            ground_truth=gt_fo,
                            predictions=pr_fo,
                        )
                    )
                    coco_result = fo.utils.eval.detection.evaluate_detections(  # type: ignore[attr-defined]
                        ds,
                        pred_field="predictions",
                        gt_field="ground_truth",
                        iou=iou_thresh,
                    )
                    coco_metrics = coco_result.metrics()
                finally:
                    ds.delete()

                print(
                    f"{name}: evaluate()={eval_score:.6f} | "
                    f"coco fscore={coco_metrics['fscore']:.6f} "
                    f"precision={coco_metrics['precision']:.6f} "
                    f"recall={coco_metrics['recall']:.6f} "
                    f"accuracy={coco_metrics['accuracy']:.6f}"
                )
