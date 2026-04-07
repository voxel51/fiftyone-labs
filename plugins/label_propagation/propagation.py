import numpy as np
import logging
import os
import uuid
from typing import Generator, Tuple, Union, Optional, Any, List
from copy import deepcopy

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.core.dataset as fod

from .sam2_local import (
    SegmentAnything2VideoModel,
    SegmentAnything2VideoModelConfig,
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


def iter_batches(
    view: fo.DatasetView,
    max_batch_size: int,
    media_mode: str,
) -> Generator[tuple[fo.DatasetView, Optional[Any]], None, None]:
    """Yields (chunk_view, overlap) for sequential batched processing with 1-element overlap.

    For image mode: overlap is the sample ID of the seam frame (None for first chunk).
    For video mode: overlap is (sample_id, frame_number) of the seam frame (None for first chunk per video).

    The caller seeds temp_input_annotation_field at the overlap element before each
    non-first chunk so SAM2 continues tracking across chunk boundaries.
    """
    if media_mode != "video":
        sample_ids = view.values("id")
        n = len(sample_ids)  # type: ignore[arg-type]
        for start in range(0, n, max_batch_size - 1):
            chunk_ids = sample_ids[start : start + max_batch_size] # type: ignore[index]
            yield view.select(chunk_ids), (chunk_ids[0] if start > 0 else None)

    else:
        for sample in view.iter_samples():
            frame_numbers = sorted(sample.frames.keys())
            n = len(frame_numbers)
            for start in range(0, n, max_batch_size - 1):
                chunk_fns = frame_numbers[start : start + max_batch_size]
                chunk_frame_ids = [sample.frames[fn].id for fn in chunk_fns]
                chunk_view = view.select([sample.id]).select_frames(
                    chunk_frame_ids
                )
                overlap = (sample.id, chunk_fns[0]) if start > 0 else None
                print(f"selected {len(chunk_frame_ids)} frames of {n}")
                yield chunk_view, overlap


def propagate_annotations_sam2(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
    max_batch_size: int = 32,
    progress: Optional[bool] = True,
) -> dict[str, float]:
    """
    Propagate annotations from exemplar frames (containing labels in input_annotation_field) to all the frames.
    Args:
        view: The view to propagate annotations from
        input_annotation_field: The field name of the annotation to copy from the exemplar frame field
        output_annotation_field: The field name of the annotation to save to the target frame
        sort_field: Field to sort samples by
        progress: Whether to show progress bars (True/False) or use default (None)
    """
    media_mode = str(view.media_type)
    if media_mode == "group":
        view = view.flatten()
        media_mode = "image"

    model = load_local_sam2(media_mode=media_mode)

    output_field = get_frame_field_name(output_annotation_field, media_mode)
    # Explicitly register the output field in the schema (needed for Teams)
    add_detection_field_if_not_exists(view._dataset, output_field)
    # Temp field: holds input_annotation_field values for all frames, and updates
    # from previous chunks' propagated outputs at overlapping frames
    # (This is to avoid overwriting the original input annotations)
    temp_input_annotation_field = (
        f"{input_annotation_field}_{os.urandom(12).hex()}"
    )
    add_detection_field_if_not_exists(
        view._dataset,
        get_frame_field_name(temp_input_annotation_field, media_mode),
    )

    run_view = (
        view.sort_by(sort_field)
        if (sort_field and view.has_field(sort_field))
        else view
    )
    run_view.set_values(
        temp_input_annotation_field, run_view.values(input_annotation_field)
    )
    try:
        for chunk_idx, (chunk_view, overlap) in enumerate(
            iter_batches(run_view, max_batch_size, media_mode)  # type: ignore[arg-type]
        ):
            if overlap is not None:
                if media_mode != "video":
                    overlap_frame = view._dataset[overlap]
                else:
                    sample_id, frame_number = overlap
                    overlap_frame = view._dataset[sample_id].frames[  # type: ignore[index]
                        frame_number
                    ]
                overlap_frame[  # type: ignore[index]
                    get_frame_field_name(
                        temp_input_annotation_field, media_mode
                    )
                ] = overlap_frame[output_field]
                overlap_frame.save()

            logger.info(f"Processing batch {chunk_idx + 1}")
            # print("\n--- before applying model")
            # if (run_view.first().frames[1]["ground_truth"] is None) or (run_view.first().frames[1]["labels_test"] is None):
            #     breakpoint()
            # print("---\n")

            # Note: `apply_model()` saves SampleViews and can drop omitted/filtered
            # content when called on frame-filtered views. To preserve all
            # existing fields, run inference on an isolated clone of the chunk
            # and only copy the output field back to the source chunk view.
            chunk_ds = chunk_view.clone(name=f"_chunk_{os.urandom(12).hex()}")
            try:
                chunk_ds.apply_model(
                    model,
                    label_field=output_field,
                    prompt_field=temp_input_annotation_field,
                    batch_size=len(chunk_ds),
                    progress=progress,
                    skip_failures=False,
                )
                chunk_view.set_values(
                    output_annotation_field,
                    chunk_ds.values(output_annotation_field),
                )
            finally:
                fo.delete_dataset(chunk_ds.name)
            # print("\n--- after applying model")
            # if (run_view.first().frames[1]["ground_truth"] is None) or (run_view.first().frames[1]["labels_test"] is None):
            #     breakpoint()
            # print("---\n")
    finally:
        delete_field_if_exists(
            view._dataset,
            get_frame_field_name(temp_input_annotation_field, media_mode),
        )

    return {}
