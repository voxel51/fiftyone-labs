import logging
from typing import Iterable, Union, Optional, Iterator, List, Tuple, Any

import numpy as np
import cv2

import fiftyone as fo
import fiftyone.core.media as fom
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


def _frame_discontinuity(img_a: np.ndarray, img_b: np.ndarray) -> bool:
    """
    Check if the two image arrays are "continuous enough".

    Args:
        img_a: First image as BGR numpy array (e.g., from cv2.imread)
        img_b: Second image as BGR numpy array

    Returns:
        True if a large discontinuity is detected between the two images,
        False otherwise
    """
    TARGET_SIZE = (256, 256)
    GRAY_CORR_THRESHOLD = 0.9
    HSV_CORR_THRESHOLD = 0.9
    GRAY_DIFF_THRESHOLD = 30

    if img_a is None or img_b is None:
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


def _compute_temporal_segments_from_frames(
    frames: Iterable[np.ndarray],
    method: str,
) -> List[Tuple[str, float]]:
    """
    Core function: compute temporal segment labels from a sequence of frames.
    Args:
        frames: Iterator of BGR image arrays (e.g., from cv2.imread or video)
        method: Segmentation method (currently "heuristic" only)
    Returns:
        List of (segment_label, exemplar_score) tuples, of length len(frames)
    """
    result: List[Tuple[str, float]] = []
    segment_count = 0

    if method == "heuristic":
        curr_segment_label = ""
        prev_frame = None
        for frame in frames:
            if prev_frame is None or _frame_discontinuity(prev_frame, frame):
                segment_count += 1
                curr_segment_label = str(fcod.ObjectId())

            result.append((curr_segment_label, 0.0))
            prev_frame = frame
    else:
        raise NotImplementedError(f"Unsupported method: {method}")
    
    logger.info(f"Computed {segment_count} temporal segments within {len(result)} frames")
    return result


def _frame_gen_from_image_dataset(
    samples: fo.core.collections.SampleCollection  # type: ignore[reportUnknownReturnType]
) -> Iterator[np.ndarray]:
    for sample in samples.iter_samples():
        frame = cv2.imread(sample.filepath)
        yield frame


def _frame_gen_from_video(
    sample: fo.Sample, max_frames: Optional[int] = None,
) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(sample.filepath)
    if not cap.isOpened():
        logger.warning(f"Failed to open video: {sample.filepath}")
        return
    try:
        frame_count = 0
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            yield frame
    finally:
        cap.release()


def extract_temporal_segments(
    view: Union[fo.Dataset, fo.DatasetView],
    method: str,
    temporal_segments_field: str,
    sort_field: Optional[str] = None,
) -> None:
    if sort_field and view.has_field(sort_field):
        view = view.sort_by(sort_field)

    media_type = view.media_type
    is_dynamic_groups = getattr(view, "_is_dynamic_groups", False)

    segment_labels: dict = {}

    if media_type == fom.IMAGE:
        labels = _compute_temporal_segments_from_frames(
            _frame_gen_from_image_dataset(view), method
        )
        id_list = view.values("id")
        for sample_id, (seg_label, exemplar_score) in zip(
            id_list, labels  # type: ignore[reportUnknownArgumentType]
        ):
            segment_labels[sample_id] = fo.Classifications(
                classifications=[
                    fo.Classification(
                        label=seg_label, exemplar_score=exemplar_score
                    )
                ]
            )
        
    elif media_type == fom.GROUP:
        if not is_dynamic_groups:
            raise NotImplementedError("Only dynamic groups are supported for grouped datasets")
        for group_view in view.iter_dynamic_groups():
            extract_temporal_segments(group_view, method, temporal_segments_field, sort_field)

    elif media_type == fom.VIDEO:
        for sample in view:
            labels = _compute_temporal_segments_from_frames(
                _frame_gen_from_video(sample, len(sample.frames)), method
            )
            
            temporal_detections: List[fo.TemporalDetection] = []
            seg_start = 1
            seg_end = 1
            prev_label = ""
            for frame_idx, (curr_label, _) in enumerate(labels):
                seg_end = frame_idx + 1
                if curr_label != prev_label:
                    if prev_label:
                        temporal_detections.append(
                            fo.TemporalDetection(
                                label=prev_label,
                                support=[seg_start, seg_end-1],
                            )
                        )
                        seg_start = seg_end + 1
                    prev_label = curr_label
            # end of iteration
            seg_end = frame_idx + 1
            if prev_label:
                temporal_detections.append(
                    fo.TemporalDetection(
                        label=prev_label,
                        support=[seg_start, seg_end],
                    )
                )

            segment_labels[sample.id] = fo.TemporalDetections(
                detections=temporal_detections
            )
    
    view.set_values(temporal_segments_field, segment_labels, key_field="id")
    view.save()


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
