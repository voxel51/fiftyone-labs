import os
import logging
from typing import Tuple, Union, Optional
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

import fiftyone as fo

logger = logging.getLogger(__name__)

# from suc_utils import evaluate_success_rate
# from embedding_utils import propagatability_pre_label, propagatability_post_label


import os
import tempfile
import shutil
from pathlib import Path
import logging
from typing import Tuple, Union, Optional, Any
import numpy as np
import cv2
from collections import OrderedDict
import urllib.request
from urllib.error import URLError

import fiftyone as fo

from .utils import bbox_corners_in_pixel_coords, fit_mask_to_bbox

logger = logging.getLogger(__name__)


def _download_sam2_weights(
    checkpoint_path: str, checkpoint_filename: str
) -> None:
    """
    Download SAM2 weights from the official repository.

    Args:
        checkpoint_path: Full path where the checkpoint should be saved
        checkpoint_filename: Name of the checkpoint file (e.g., "sam2.1_hiera_tiny.pt")
    Raises:
        RuntimeError: If download fails
    """
    weights_url = f"https://dl.fbaipublicfiles.com/segment_anything_2/{checkpoint_filename}"

    logger.info(f"Downloading SAM2 weights from {weights_url}...")
    logger.info(f"Target location: {checkpoint_path}")

    # Create weights directory if it doesn't exist
    weights_dir = os.path.dirname(checkpoint_path)
    os.makedirs(weights_dir, exist_ok=True)

    try:
        # Download with progress reporting
        def _show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(
                    100, (block_num * block_size * 100) // total_size
                )
                if percent % 10 == 0:  # Log every 10%
                    logger.info(f"Download progress: {percent}%")

        urllib.request.urlretrieve(
            weights_url, checkpoint_path, _show_progress
        )
        logger.info(f"Successfully downloaded weights to {checkpoint_path}")

    except URLError as e:
        raise RuntimeError(
            f"Failed to download SAM2 weights from {weights_url}: {e}. "
            f"Please download manually from https://github.com/facebookresearch/segment-anything-2 "
            f"and place {checkpoint_filename} in {weights_dir}/"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error downloading SAM2 weights: {e}")

    # Verify file was downloaded (check file size > 0)
    if (
        not os.path.exists(checkpoint_path)
        or os.path.getsize(checkpoint_path) == 0
    ):
        raise RuntimeError(
            f"SAM2 weights file {checkpoint_filename} is empty or does not exist:"
            f"Please download manually from https://github.com/facebookresearch/segment-anything-2 "
            f"and place {checkpoint_filename} in {weights_dir}/"
        )


class PropagatorSAM2:
    def __init__(self, model_cfg=None, checkpoint=None):
        """
        Initialize SAM2 propagator.
        1. Install SAM2 from https://github.com/facebookresearch/segment-anything-2
        2. Download the config and checkpoint to the installed location under weights/
        """
        # SAM2 uses Hydra config loading, which expects config names relative to its search path
        # Use the config name that exists in the SAM2 package, not an absolute path
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.checkpoint = "weights/sam2.1_hiera_tiny.pt"
        self.sam2_predictor: Any = None  # type: ignore[assignment]
        self.inference_state: Any = None  # type: ignore[assignment]
        self.setup()
        self.preds_dict = OrderedDict()
        self.label_type = "bounding_box"

    def setup(self):
        import torch

        device = torch.device(
            # "mps" if torch.backends.mps.is_available() else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            # )  # avoid MPS to prevent Metal SIGABRTs
        )

        try:
            import sam2
        except ImportError:
            RuntimeError(
                "SAM2 is not installed. Please install it with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )
        from sam2.build_sam import build_sam2_video_predictor

        package_dir = os.path.dirname(os.path.abspath(sam2.__file__))
        checkpoint_path = os.path.join(package_dir, self.checkpoint)

        # Check if weights exist, if not try to download them
        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"SAM2 checkpoint not found at {checkpoint_path}. Attempting to download..."
            )
            checkpoint_filename = os.path.basename(self.checkpoint)
            _download_sam2_weights(checkpoint_path, checkpoint_filename)

        self.sam2_predictor = build_sam2_video_predictor(
            self.model_cfg, checkpoint_path, device=str(device)
        )
        logger.info("SAM2 predictor initialized successfully")

    def path_list_to_dir(self, image_path_list):
        """
        Convert a list of image paths to a temporary directory
        using simlinks, maintaining the order of the images.

        Args:
            image_path_list: List of image file paths
        Returns:
            Temporary directory path
        """
        tmpdir = Path(tempfile.mkdtemp())
        for ii, pp in enumerate(image_path_list):
            tmp_path = tmpdir / f"{ii:06d}{Path(pp).suffix}"
            tmp_path.symlink_to(Path(pp).resolve())
        logger.info(
            f"Created temporary directory {tmpdir} with {len(image_path_list)} frames"
        )
        return tmpdir

    def initialize(self, frame_path_list):
        """
        Initialize the inference state with the frames list.
        Args:
            frame_path_list: List of frame file paths ordered by frame number
        """
        # TODO(neeraja): handle video files
        frames_dir = self.path_list_to_dir(frame_path_list)
        self.inference_state = self.sam2_predictor.init_state(str(frames_dir))
        self.preds_dict.clear()
        for idx, frame_path in enumerate(frame_path_list):
            self.preds_dict[os.path.abspath(frame_path)] = None
        shutil.rmtree(frames_dir)
        logger.info(
            f"Inference state initialized with {len(frame_path_list)} frames; cleaned up temporary directory {frames_dir}"
        )

    def extract_spatial_embeddings(self, frame_filepath, feature_level=0):
        """
        Extract patch-wise embeddings for all images in the inference state.

        After initialize() (stage 1), this extracts vision features for all frames.
        The embeddings are returned with spatial dimensions preserved.

        Args:
            frame_filepath: The frame file path to extract embeddings for
            feature_level: Which feature level to extract (0=highest res (default), -1=lowest res)

        Returns:
            embedding: A tensor with shape (C, H, W) for the given feature level.
            C = number of channels, H = embedding height, W = embedding width
        """
        import torch

        if self.inference_state is None:
            raise RuntimeError(
                "Must call initialize() before extract_spatial_embeddings()"
            )

        frame_idx = list(self.preds_dict.keys()).index(
            os.path.abspath(frame_filepath)
        )
        logger.debug(
            f"Extracting patch embeddings for frame {frame_filepath}..."
        )

        with torch.no_grad():
            (
                _,
                _,
                vision_feats,
                _,
                feat_sizes,
            ) = self.sam2_predictor._get_image_feature(
                self.inference_state,
                frame_idx,
                batch_size=1,
            )
            vision_feat = vision_feats[feature_level]
            feat_size = feat_sizes[feature_level]

            # feat shape: (HW, B, C) -> reshape to (B, C, H, W)
            H, W = feat_size
            B, C = vision_feat.shape[1], vision_feat.shape[2]
            spatial_feat = vision_feat.permute(1, 2, 0).view(B, C, H, W)
            spatial_feat = spatial_feat.squeeze(0).cpu().numpy()

        logger.debug(f"Extracted spatial embeddings for {frame_filepath}")
        return spatial_feat

    def register_source_frame(self, source_filepath, source_detections):
        """
        Register the source frame and detections with SAM2.

        Args:
            source_filepath: The source frame file path
            source_detections: The detections from source_frame (fo.Detections)
        """
        cv2.setNumThreads(1)

        if not hasattr(source_detections, "detections"):
            logger.warning(
                f"Source detections is either empty, or not a fo.Detections object: {source_detections}"
            )
            return

        source_frame_idx = list(self.preds_dict.keys()).index(
            os.path.abspath(source_filepath)
        )
        logger.debug(
            f"Registering source frame {source_filepath} at index {source_frame_idx}"
        )

        source_frame = cv2.imread(source_filepath)
        source_height, source_width = source_frame.shape[:2]

        for detection in source_detections.detections:
            # Get source bbox and convert to pixel coordinates
            source_bbox = detection.bounding_box
            x1, y1, x2, y2 = bbox_corners_in_pixel_coords(
                source_bbox, source_width, source_height
            )

            source_mask = detection.mask
            if source_mask is not None:
                self.label_type = "segmentation_mask"
                source_mask_fitted = fit_mask_to_bbox(
                    source_mask, (y2 - y1, x2 - x1)
                )
                # make it relative to the target frame
                source_mask_framed = np.zeros(
                    (source_height, source_width), bool
                )
                source_mask_framed[y1:y2, x1:x2] = source_mask_fitted
                source_mask_framed = source_mask_framed.astype(np.uint8)

                self.sam2_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=source_frame_idx,
                    obj_id=detection.label,
                    mask=source_mask_framed,
                )
                logger.debug(f"Added new segmentation mask: {detection.label}")
            else:
                self.sam2_predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=source_frame_idx,
                    obj_id=detection.label,
                    box=[x1, y1, x2, y2],
                )
                logger.debug(f"Added new bounding box: {detection.label}")

    def propagate_to_all_frames(self):
        """
        Propagate detections to all frames in the inference state.
        Uses SAM2's propagate_in_video API.
        Returns:
            None
            Populates the preds_dict with predictions for each frame
        """
        cv2.setNumThreads(1)

        propagated_to_count = 0
        for frame_idx, obj_ids, mask_logits in self.sam2_predictor.propagate_in_video(self.inference_state):  # type: ignore[union-attr]
            target_filepath = list(self.preds_dict.keys())[frame_idx]
            target_frame = cv2.imread(target_filepath)
            target_height, target_width = target_frame.shape[:2]

            propagated_detections = []
            obj_ids = list(obj_ids)
            logger.debug(
                f"Found {len(obj_ids)} detections for frame {target_filepath}"
            )
            for oi, obj_id in enumerate(obj_ids):
                logits = mask_logits[oi].squeeze(0)
                pred = (logits > 0).cpu().numpy().astype(
                    np.uint8
                ) * 255  # threshold at 0

                # Find new bbox from pred
                contours, _ = cv2.findContours(
                    pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    new_bbox = [
                        x / target_width,
                        y / target_height,
                        w / target_width,
                        h / target_height,
                    ]
                else:
                    logger.warning("Warning: No contour found for detection")
                    new_bbox = (0, 0, 0, 0)

                if self.label_type == "segmentation_mask":
                    (
                        x1_new,
                        y1_new,
                        x2_new,
                        y2_new,
                    ) = bbox_corners_in_pixel_coords(
                        new_bbox, target_width, target_height
                    )
                    mask_fitted = pred[y1_new:y2_new, x1_new:x2_new]
                    mask_fitted = (
                        mask_fitted.astype(np.float32) / np.max(mask_fitted)
                    ).astype(np.uint8)
                else:
                    mask_fitted = None

                new_detection = fo.Detection(
                    bounding_box=new_bbox,
                    mask=mask_fitted,
                    label=obj_id,
                )
                propagated_detections.append(new_detection)

            logger.debug(
                f"Propagated {len(propagated_detections)} detections for frame {target_filepath}"
            )
            self.preds_dict[target_filepath] = fo.Detections(
                detections=propagated_detections
            )
            propagated_to_count += 1

        logger.info(
            f"Propagated detections to {propagated_to_count} frames out of {len(self.preds_dict)}"
        )

    def get_propagated_detections(self, target_filepath) -> fo.Detections:
        """
        Get propagated detections for target frame.
        This does not modify the inference state.

        Args:
            target_filepath: Target frame file path
        Returns:
            fo.Detections: Propagated detections with or without masks
        """
        if self.inference_state is None:
            raise RuntimeError(
                "Must call register_source_frame() before get_propagated_detections()"
            )

        if os.path.abspath(target_filepath) not in self.preds_dict:
            logger.warning(
                f"Target frame {target_filepath} not found in predictions"
            )
            return fo.Detections(detections=[])

        result = self.preds_dict.get(
            os.path.abspath(target_filepath), fo.Detections(detections=[])
        )
        return result


def propagate_annotations_sam2(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: str,
    progress: Optional[bool] = True,
) -> dict[str, float]:
    """
    Propagate annotations from exemplar frames (containing labels in input_annotation_field) to all the frames.
    Args:
        view: The view to propagate annotations from
        input_annotation_field: The field name of the annotation to copy from the exemplar frame field
        output_annotation_field: The field name of the annotation to save to the target frame
        evaluate_propagation: Whether to evaluate the propagation against
                              the input annotation field present in the propagation targets.
        sort_field: Field to sort samples by
        progress: Whether to show progress bars (True/False) or use default (None)
    """

    # Set up the propagator
    propagator = PropagatorSAM2()

    if view.has_field(sort_field):
        image_path_list = view.sort_by(sort_field).values("filepath")
    else:
        image_path_list = view.values("filepath")
    propagator.initialize(image_path_list)

    # Register all frames
    def register_sample(sample):
        if sample[input_annotation_field]:
            propagator.register_source_frame(
                sample.filepath, sample[input_annotation_field]
            )

    _ = list(
        view.map_samples(register_sample, num_workers=1, progress=progress)
    )

    # Propagate
    try:
        propagator.propagate_to_all_frames()
    except Exception as e:
        """
        ------------------------------------------------------------
        This error typically occurs due to a mixed-precision dtype mismatch in SAM2's
        internal transformer layers. SAM2 uses bfloat16 for some tensors while keeping
        others (or layer weights) in float32, causing "mat1 and mat2 must have the same
        dtype, but got BFloat16 and Float" errors in Linear layers during propagation.

        This is a known issue in SAM2's implementation that is more likely to occur
        when propagating across multiple sequences with different label sets,
        as this triggers more complex memory attention paths in SAM2's tracking stack.
        ------------------------------------------------------------
        """
        raise RuntimeError(
            f"Error propagating to all frames; \
        please try with a shorter sequence with continuous frames and consistent labels."
        )

    # Populate propagations
    def populate_propagations(sample):
        propagated_detections = propagator.get_propagated_detections(
            sample.filepath
        )
        sample[output_annotation_field] = propagated_detections
        return

    if view.has_field(sort_field):
        _ = list(
            view.sort_by(sort_field).map_samples(
                populate_propagations,
                num_workers=1,
                save=True,
                progress=progress,
            )
        )
    else:
        _ = list(
            view.map_samples(
                populate_propagations,
                num_workers=1,
                save=True,
                progress=progress,
            )
        )

    return {}
