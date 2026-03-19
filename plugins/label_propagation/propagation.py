import numpy as np
import logging
from typing import Tuple, Union, Optional, Any, List
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


def propagate_annotations_sam2(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
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

    run_view = (
        view.sort_by(sort_field)
        if (sort_field and view.has_field(sort_field))
        else view
    )
    run_view.apply_model(
        model,
        # label_field is applied directly to the frame-level field
        label_field=output_field,
        prompt_field=input_annotation_field,
        batch_size=int(2 ** np.ceil(np.log2(len(run_view)))),  # type: ignore[arg-type]
        progress=progress,
        skip_failures=False,
    )

    return {}
