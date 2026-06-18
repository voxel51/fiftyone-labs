import logging
import os
from typing import Generator, Tuple, Union, Optional, Any, List, Literal
from copy import deepcopy
from contextlib import contextmanager
from functools import partial
import numpy as np

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.core.dataset as fod
import fiftyone.core.labels as fol

from .sam2_local import (
    SegmentAnything2VideoModel,
    SegmentAnything2VideoModelConfig,
    to_abs_mask,
    logits_to_box_and_mask,
    detection_to_abs_box_xyxy,
)

logger = logging.getLogger(__name__)


SUPPORTED_PROPAGATION_METHODS = [
    "sam2",
]

_SAM2_ZOO_MODEL_NAME = "segment-anything-2-hiera-tiny-video-torch"
_SAM2_LOCAL_MODEL_CACHE: dict[str, Any] = {}


def load_local_sam2(media_mode: str):
    if media_mode in _SAM2_LOCAL_MODEL_CACHE:
        return _SAM2_LOCAL_MODEL_CACHE[media_mode]

    foz.ensure_zoo_model_requirements(
        _SAM2_ZOO_MODEL_NAME, error_level=None, log_success=False
    )
    zoo_model, model_path = foz.download_zoo_model(_SAM2_ZOO_MODEL_NAME)

    config_dict = deepcopy(zoo_model.default_deployment_config_dict)
    inner = config_dict.setdefault("config", {})
    inner["media_mode"] = media_mode
    inner["model_path"] = model_path
    inner[
        "model_name"
    ] = None  # prevent zoo re-download; fom.load_model sets this too

    config = SegmentAnything2VideoModelConfig(inner)
    model = SegmentAnything2VideoModel(config)
    _SAM2_LOCAL_MODEL_CACHE[media_mode] = model

    return model


def get_frame_field_name(field_name: str, media_mode: str) -> str:
    if media_mode == "video":
        if field_name.startswith("frames."):
            return field_name[len("frames.") :]
    return field_name


def add_detection_field_if_not_exists(dataset: fo.Dataset, field_name: str):
    if str(dataset.media_type) == "video":
        if field_name not in dataset.get_frame_field_schema():  # type: ignore[arg-type]
            dataset.add_frame_field(
                field_name,
                fo.EmbeddedDocumentField,
                embedded_doc_type=fo.Detections,
            )
    else:
        if field_name not in dataset.get_field_schema():
            dataset.add_sample_field(
                field_name,
                fo.EmbeddedDocumentField,
                embedded_doc_type=fo.Detections,
            )

    # Reload the dataset singleton (if any)
    # needed because the Teams executor runs operators with no_singleton_cache=True
    singleton = fod.Dataset._instances.get(dataset.name)  # type: ignore[attr-defined]
    if singleton is not None and singleton is not dataset:
        singleton.reload()


def delete_field_if_exists(dataset: fo.Dataset, field_name: str):
    if str(dataset.media_type) == "video":
        if field_name in dataset.get_frame_field_schema():  # type: ignore[arg-type]
            dataset.delete_frame_field(field_name)
    else:
        if field_name in dataset.get_field_schema():
            dataset.delete_sample_field(field_name)

    # Reload the dataset singleton (if any)
    # needed because the Teams executor runs operators with no_singleton_cache=True
    singleton = fod.Dataset._instances.get(dataset.name)  # type: ignore[attr-defined]
    if singleton is not None and singleton is not dataset:
        singleton.reload()


def iter_image_batches(
    view: fo.DatasetView,
    batch_size: int,
    direction: Literal["forward", "backward"] = "forward",
) -> Generator[tuple[fo.DatasetView, Optional[Any]], None, None]:
    """Yields (chunk_view, overlap) for sequential batched processing with 1-element overlap.

    overlap is the sample ID of the seam frame (None for first chunk).

    The caller seeds temp_input_annotation_field at the overlap element before each
    non-first chunk so SAM2 continues tracking across chunk boundaries.
    """
    sample_ids = view.values("id")
    n = len(sample_ids)  # type: ignore[arg-type]
    if direction == "forward":
        for start in range(0, n, batch_size - 1):
            chunk_ids = sample_ids[start : start + batch_size]  # type: ignore[index]
            yield view.select(chunk_ids, ordered=True), (
                chunk_ids[0] if start > 0 else None
            )
    else:
        for start in range(n - 1, -1, -batch_size + 1):
            chunk_ids = sample_ids[max(start - batch_size + 1, 0) : start + 1]  # type: ignore[index]
            yield view.select(chunk_ids, ordered=True), (
                chunk_ids[-1] if start < n - 1 else None
            )


@contextmanager
def sam2_chunk_direction(
    model: SegmentAnything2VideoModel,
    direction: Literal["forward", "backward"],
):
    """
    Context manager to set the propagate_in_reverse attribute
    of the SegmentAnything2VideoModel to the direction of the chunk.
    """
    had = hasattr(model, "propagate_in_reverse")
    prev = getattr(model, "propagate_in_reverse", False)
    model.propagate_in_reverse = bool(direction == "backward")  # type: ignore[attr-defined]
    try:
        yield
    finally:
        if had:
            model.propagate_in_reverse = prev  # type: ignore[attr-defined]
        elif hasattr(model, "propagate_in_reverse"):
            delattr(model, "propagate_in_reverse")


def _fuse_bounding_boxes(*boxes_xyxy, width, height):
    b = np.stack(boxes_xyxy)
    mn, mx = b[:, :2].min(0), b[:, 2:4].max(0)
    return [
        mn[0] / width,
        mn[1] / height,
        (mx[0] - mn[0]) / width,
        (mx[1] - mn[1]) / height,
    ]


def _fuse_nonempty_detections(fdet, bdet, width, height):
    fbox = detection_to_abs_box_xyxy(fdet, width, height)
    bbox = detection_to_abs_box_xyxy(bdet, width, height)
    if fdet.mask is not None and bdet.mask is not None:
        fused_mask_array = np.maximum(
            to_abs_mask(fdet.mask, fbox, width, height),
            to_abs_mask(bdet.mask, bbox, width, height),
        )
        fused_box, fused_mask = logits_to_box_and_mask(
            fused_mask_array, width, height
        )
        if fused_box is None:
            return None
    else:
        fused_box = _fuse_bounding_boxes(
            fbox, bbox, width=width, height=height
        )
        fused_mask = None
    return fol.Detection(
        label=fdet.label,
        bounding_box=fused_box,
        mask=fused_mask,
        index=fdet.index,
    )


def _fuse_forward_backward_outputs(
    sample: Union[fo.Sample, fo.core.frame.FrameView],  # type: ignore[arg-type]
    forward_output_field: str,
    backward_output_field: str,
    output_field: str,
    sample_width: int,
    sample_height: int,
):
    if hasattr(sample, "metadata") and sample.metadata is not None:
        sample_width = sample.metadata.width or sample_width
        sample_height = sample.metadata.height or sample_height
    forward_detections = sample.get_field(forward_output_field)
    backward_detections = sample.get_field(backward_output_field)
    if (
        (forward_detections is None)
        or (forward_detections.detections is None)
        or (len(forward_detections.detections) == 0)
    ):
        fused_detections = backward_detections.detections
    elif (
        (backward_detections is None)
        or (backward_detections.detections is None)
        or (len(backward_detections.detections) == 0)
    ):
        fused_detections = forward_detections.detections
    else:
        # match objects of the same index from the forward and backward outputs
        fused_detections = []
        fdet_by_index = {d.index: d for d in forward_detections.detections}
        bdet_by_index = {d.index: d for d in backward_detections.detections}
        for index in fdet_by_index.keys() | bdet_by_index.keys():
            fdet = fdet_by_index.get(index)
            bdet = bdet_by_index.get(index)
            if fdet is None:
                fused_detections.append(bdet)
                continue
            if bdet is None:
                fused_detections.append(fdet)
                continue

            fused = _fuse_nonempty_detections(
                fdet, bdet, sample_width, sample_height
            )
            if fused is not None:
                fused_detections.append(fused)

    sample.set_field(output_field, fol.Detections(detections=fused_detections))
    sample.save()


def _fuse_forward_backward_outputs_video_sample(
    sample: fo.Sample,
    forward_output_field: str,
    backward_output_field: str,
    output_field: str,
    sample_width: int,
    sample_height: int,
) -> None:
    if hasattr(sample, "metadata") and sample.metadata is not None:
        sample_width = sample.metadata.frame_width or sample_width
        sample_height = sample.metadata.frame_height or sample_height
    for frame in sample.frames.values():
        _fuse_forward_backward_outputs(
            frame,
            forward_output_field,
            backward_output_field,
            output_field,
            sample_width,
            sample_height,
        )


def _copy_forward_outputs(
    sample: fo.Sample,
    forward_output_field: str,
    output_field: str,
) -> None:
    sample.set_field(
        output_field, deepcopy(sample.get_field(forward_output_field))
    )
    sample.save()


def _copy_forward_outputs_video_sample(
    sample: fo.Sample,
    forward_output_field: str,
    output_field: str,
) -> None:
    for frame in sample.frames.values():
        frame.set_field(
            output_field, deepcopy(frame.get_field(forward_output_field))
        )


def propagate_annotations_sam2(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
    batch_size: int = 32,
    progress: Optional[bool] = True,
    bidirectional: bool = True,
) -> dict[str, float]:
    """
    Propagate annotations from exemplar frames (containing labels in input_annotation_field) to all the frames.
    Args:
        view: The view to propagate annotations from
        input_annotation_field: The field name of the annotation to copy from the exemplar frame field
        output_annotation_field: The field name of the annotation to save to the target frame
        sort_field: Field to sort samples by
        progress: Whether to show progress bars (True/False) or use default (None)
        bidirectional: If True, run forward and backward passes and fuse
    """
    media_mode = str(view.media_type)
    if media_mode == "group":
        view = view.flatten()
        media_mode = "image"

    model = load_local_sam2(media_mode=media_mode)

    output_field = get_frame_field_name(output_annotation_field, media_mode)
    # Explicitly register the output field in the schema (needed for Teams)
    add_detection_field_if_not_exists(view._dataset, output_field)

    run_view = (
        view.sort_by(sort_field)
        if (sort_field and view.has_field(sort_field))
        else view
    )

    random_suffix = os.urandom(12).hex()

    # For bidirectional propagation, we create temp fields for the
    # forward and backward pass outputs, before fusing them.
    temp_output_field_fwd = f"{output_field}_{random_suffix}_fwd"
    temp_output_field_bwd = f"{output_field}_{random_suffix}_bwd"
    add_detection_field_if_not_exists(
        view._dataset,
        temp_output_field_fwd,
    )
    if bidirectional:
        add_detection_field_if_not_exists(
            view._dataset,
            temp_output_field_bwd,
        )

    if media_mode == "video":
        try:
            with sam2_chunk_direction(model, "forward"):
                run_view.apply_model(
                    model,
                    # label_field is applied directly to the frame-level field
                    label_field=temp_output_field_fwd,
                    prompt_field=input_annotation_field,
                    batch_size=batch_size,
                    progress=progress,
                    skip_failures=False,
                )
            if bidirectional:
                with sam2_chunk_direction(model, "backward"):
                    run_view.apply_model(
                        model,
                        label_field=temp_output_field_bwd,
                        prompt_field=input_annotation_field,
                        batch_size=batch_size,
                        progress=progress,
                        skip_failures=False,
                    )
                _fuse_kw = dict(
                    forward_output_field=temp_output_field_fwd,
                    backward_output_field=temp_output_field_bwd,
                    output_field=output_field,
                    sample_width=model._curr_frame_width,
                    sample_height=model._curr_frame_height,
                )
                for _ in run_view.map_samples(
                    partial(
                        _fuse_forward_backward_outputs_video_sample, **_fuse_kw
                    ),
                    save=True,
                    progress=progress,
                    num_workers=1,
                ):
                    pass
            else:
                for _ in run_view.map_samples(
                    partial(
                        _copy_forward_outputs_video_sample,
                        forward_output_field=temp_output_field_fwd,
                        output_field=output_field,
                    ),
                    save=True,
                    progress=progress,
                    num_workers=1,
                ):
                    pass
        finally:
            for fn in (
                temp_output_field_fwd,
                temp_output_field_bwd,
            ):
                delete_field_if_exists(view._dataset, fn)
        return {}

    # For images, we support batching by chunking the view into batch_size images.
    # We create a temp field to hold the input_annotation_field values for all frames
    # and updates from previous chunks' propagated outputs at overlapping frames.
    # This is to avoid overwriting the original input annotations.
    temp_input_annotation_field_fwd = (
        f"{input_annotation_field}_{random_suffix}_fwd"
    )
    temp_input_annotation_field_bwd = (
        f"{input_annotation_field}_{random_suffix}_bwd"
    )
    add_detection_field_if_not_exists(
        view._dataset,
        temp_input_annotation_field_fwd,
    )
    run_view.set_values(
        temp_input_annotation_field_fwd,
        run_view.values(input_annotation_field),
    )
    if bidirectional:
        add_detection_field_if_not_exists(
            view._dataset,
            temp_input_annotation_field_bwd,
        )
        run_view.set_values(
            temp_input_annotation_field_bwd,
            run_view.values(input_annotation_field),
        )

    try:
        for chunk_idx, (chunk_view, overlap) in enumerate(
            iter_image_batches(run_view, batch_size)  # type: ignore[arg-type]
        ):
            if overlap is not None:
                overlap_frame = view._dataset[overlap]
                overlap_frame[  # type: ignore[index]
                    temp_input_annotation_field_fwd
                ] = overlap_frame[temp_output_field_fwd]
                overlap_frame.save()

            logger.info(f"Processing forward batch {chunk_idx + 1}")

            with sam2_chunk_direction(model, "forward"):
                chunk_view.apply_model(
                    model,
                    label_field=temp_output_field_fwd,
                    prompt_field=temp_input_annotation_field_fwd,
                    batch_size=batch_size,
                    progress=progress,
                    skip_failures=False,
                )

        if bidirectional:
            for chunk_idx, (chunk_view, overlap) in enumerate(
                iter_image_batches(run_view, batch_size, direction="backward")  # type: ignore[arg-type]
            ):
                if overlap is not None:
                    overlap_frame = view._dataset[overlap]
                    overlap_frame[  # type: ignore[index]
                        temp_input_annotation_field_bwd
                    ] = overlap_frame[temp_output_field_bwd]
                    overlap_frame.save()

                logger.info(f"Processing backward batch {chunk_idx + 1}")

                with sam2_chunk_direction(model, "backward"):
                    chunk_view.apply_model(
                        model,
                        label_field=temp_output_field_bwd,
                        prompt_field=temp_input_annotation_field_bwd,
                        batch_size=batch_size,
                        progress=progress,
                        skip_failures=False,
                    )
                    # TODO(neeraja)(low-priority): Ensure that the backward pass stays
                    # strictly backward inside the last chunk

        _fuse_kw = dict(
            forward_output_field=temp_output_field_fwd,
            backward_output_field=temp_output_field_bwd,
            output_field=output_field,
            sample_width=model._curr_frame_width,
            sample_height=model._curr_frame_height,
        )
        if bidirectional:
            for _ in run_view.map_samples(
                partial(_fuse_forward_backward_outputs, **_fuse_kw),
                save=True,
                progress=progress,
                num_workers=1,
            ):
                pass
        else:
            for _ in run_view.map_samples(
                partial(
                    _copy_forward_outputs,
                    forward_output_field=temp_output_field_fwd,
                    output_field=output_field,
                ),
                save=True,
                progress=progress,
                num_workers=1,
            ):
                pass
    finally:
        for fn in (
            temp_input_annotation_field_fwd,
            temp_input_annotation_field_bwd,
            temp_output_field_fwd,
            temp_output_field_bwd,
        ):
            delete_field_if_exists(view._dataset, fn)

    return {}
