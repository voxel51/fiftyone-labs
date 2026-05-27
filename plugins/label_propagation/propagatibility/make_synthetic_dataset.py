"""
Build a synthetic FiftyOne image dataset for label-propagation experiments.

Four sequences of 15 frames each (``frame_number`` 0–14), with a blue
background and a yellow circle; per-sequence motion/color/occlusion varies.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

import fiftyone as fo

_PLUGINS_DIR = Path(__file__).resolve().parent.parent
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

from label_propagation.embedding_utils import (  # type: ignore
    get_sam2_embeddings,
)


logger = logging.getLogger(__name__)

IMAGE_SIZE = 256
NUM_FRAMES = 15

SEQUENCE_IDS = (
    "move_right",
    "move_in",
    "fade_color",
    "partly_occlude",
)

BG_RGB = (73, 156, 239)
CIRCLE_RGB = (255, 235, 82)
CIRCLE_RGB_LIGHT = (255, 253, 238)
OCCLUDER_RGB = (89, 166, 92)

CIRCLE_RADIUS = 20
CENTER_X, CENTER_Y = 128, 128
STEP_PX = 4
OCCLUDER_WIDTH = 8
OCCLUDER_STEP_PX = 8

CIRCLE_LABEL = "circle"


def _lerp_color(
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    return tuple(int(s + t * (e - s)) for s, e in zip(start, end))


def _circle_in_frame(cx: int, cy: int, radius: int, size: int) -> bool:
    return (
        cx + radius > 0
        and cy + radius > 0
        and cx - radius < size
        and cy - radius < size
    )


def _circle_position(
    sequence_id: str, frame_number: int
) -> Optional[Tuple[int, int]]:
    """Return circle center in pixel coords, or None if not drawn this frame."""
    if sequence_id == "move_right":
        return (CENTER_X + frame_number * STEP_PX, CENTER_Y)

    if sequence_id == "move_in":
        # Reverse of the old move_out timeline: circle enters from off-frame
        # top-left (+4 px right/down per frame) and ends near (25, 25).
        rev_frame = NUM_FRAMES - 1 - frame_number
        cx = 25 - rev_frame * STEP_PX
        cy = 25 - rev_frame * STEP_PX
        if _circle_in_frame(cx, cy, CIRCLE_RADIUS, IMAGE_SIZE):
            return (cx, cy)
        return None

    if sequence_id in ("fade_color", "partly_occlude"):
        return (CENTER_X, CENTER_Y)

    raise ValueError(f"Unknown sequence_id: {sequence_id}")


def _circle_color(sequence_id: str, frame_number: int) -> Tuple[int, int, int]:
    if sequence_id == "fade_color":
        t = frame_number / (NUM_FRAMES - 1) if NUM_FRAMES > 1 else 0.0
        return _lerp_color(CIRCLE_RGB, CIRCLE_RGB_LIGHT, t)
    return CIRCLE_RGB


def _circle_bounding_box(cx: int, cy: int, radius: int) -> List[float]:
    """Relative [x, y, w, h] bounding box for the circle."""
    xmin = cx - radius
    ymin = cy - radius
    side = 2 * radius
    return [
        xmin / IMAGE_SIZE,
        ymin / IMAGE_SIZE,
        side / IMAGE_SIZE,
        side / IMAGE_SIZE,
    ]


def ground_truth_for_frame(
    sequence_id: str, frame_number: int
) -> fo.Detections:
    """Build ground-truth detections for one frame."""
    position = _circle_position(sequence_id, frame_number)
    if position is None:
        return fo.Detections(detections=[])

    cx, cy = position
    return fo.Detections(
        detections=[
            fo.Detection(
                label=CIRCLE_LABEL,
                bounding_box=_circle_bounding_box(cx, cy, CIRCLE_RADIUS),
            )
        ]
    )


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    radius: int,
    fill: Tuple[int, int, int],
) -> None:
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=fill,
    )


def _draw_vertical_bar(
    draw: ImageDraw.ImageDraw,
    bar_left: int,
    width: int,
    size: int,
    fill: Tuple[int, int, int],
) -> None:
    draw.rectangle((bar_left, 0, bar_left + width - 1, size - 1), fill=fill)


def render_sequence_frame(sequence_id: str, frame_number: int) -> Image.Image:
    """Render one synthetic frame for the given sequence and frame index."""
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BG_RGB)
    draw = ImageDraw.Draw(img)

    position = _circle_position(sequence_id, frame_number)
    if position is not None:
        cx, cy = position
        _draw_circle(
            draw,
            cx,
            cy,
            CIRCLE_RADIUS,
            _circle_color(sequence_id, frame_number),
        )

    if sequence_id == "partly_occlude":
        circle_left = CENTER_X - CIRCLE_RADIUS
        start_bar_left = circle_left - OCCLUDER_WIDTH - OCCLUDER_STEP_PX
        bar_left = start_bar_left + frame_number * OCCLUDER_STEP_PX
        _draw_vertical_bar(
            draw, bar_left, OCCLUDER_WIDTH, IMAGE_SIZE, OCCLUDER_RGB
        )

    return img


def write_sequence_images(output_dir: str) -> List[dict]:
    """Write all sequence frames to disk and return sample metadata dicts."""
    sample_infos: List[dict] = []
    for sequence_id in SEQUENCE_IDS:
        seq_dir = os.path.join(output_dir, sequence_id)
        os.makedirs(seq_dir, exist_ok=True)
        for frame_number in range(NUM_FRAMES):
            img = render_sequence_frame(sequence_id, frame_number)
            filename = f"{frame_number:05d}.jpg"
            filepath = os.path.join(seq_dir, filename)
            img.save(filepath, quality=95)
            sample_infos.append(
                {
                    "filepath": filepath,
                    "sequence_id": sequence_id,
                    "frame_number": frame_number,
                    "ground_truth": ground_truth_for_frame(
                        sequence_id, frame_number
                    ),
                }
            )
    return sample_infos


def make_synthetic_dataset(
    dataset_name: str = "synthetic_label_prop",
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> fo.Dataset:
    """Create the synthetic dataset and return it.

    Args:
        dataset_name: FiftyOne dataset name.
        output_dir: Directory for JPEG frames. Defaults to a folder next to
            this script named ``synthetic_label_prop_images``.
        overwrite: If True, delete an existing dataset with the same name and
            replace on-disk images under ``output_dir``.

    Returns:
        the :class:`fiftyone.core.dataset.Dataset`
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "synthetic_label_prop_images",
        )
    output_dir = os.path.abspath(output_dir)

    if fo.dataset_exists(dataset_name):
        if overwrite:
            fo.delete_dataset(dataset_name)
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' already exists; pass overwrite=True"
            )

    if overwrite and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    sample_infos = write_sequence_images(output_dir)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    samples = [
        fo.Sample(
            filepath=info["filepath"],
            sequence_id=info["sequence_id"],
            frame_number=info["frame_number"],
            ground_truth=info["ground_truth"],
        )
        for info in sample_infos
    ]
    dataset.add_samples(samples)
    dataset.compute_metadata()

    logger.info(
        "Created dataset '%s' with %d samples in %s",
        dataset_name,
        len(samples),
        output_dir,
    )

    logger.info("Computing SAM2 embeddings...")
    _ = get_sam2_embeddings(dataset)

    return dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic FiftyOne image dataset for label propagation.",
    )
    parser.add_argument(
        "--dataset-name",
        default="synthetic_label_prop",
        help="FiftyOne dataset name (default: synthetic_label_prop)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for rendered JPEGs (default: synthetic_label_prop_images/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing dataset and image files",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the dataset",
    )
    return parser.parse_args()


def display_dataset(dataset: fo.Dataset):
    session = fo.launch_app(dataset)
    input("Press Enter to close the app...")
    session.close()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    dataset = None
    if not args.overwrite and fo.dataset_exists(args.dataset_name):
        print(
            f"Dataset '{args.dataset_name}' already exists; "
            "pass --overwrite to replace"
        )
        dataset = fo.load_dataset(args.dataset_name)
    else:
        dataset = make_synthetic_dataset(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(f"Dataset '{dataset.name}' created with {len(dataset)} samples")
        print(f"  media_type: {dataset.media_type}")
        print(f"  sequences: {dataset.distinct('sequence_id')}")
        print(f"  frame_number range: {dataset.bounds('frame_number')}")

    if dataset is not None and args.display:
        display_dataset(dataset)
