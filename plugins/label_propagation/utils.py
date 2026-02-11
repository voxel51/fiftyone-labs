from typing import Tuple, Union, Optional
import numpy as np
import cv2


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


def bbox_corners_in_pixel_coords(bbox, image_width, image_height):
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
