import numpy as np
import logging
from typing import Tuple, Union, Optional, Any, List
from copy import deepcopy

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.core.models as fom

from .sam2_local import SegmentAnything2VideoModel

logger = logging.getLogger(__name__)


SUPPORTED_PROPAGATION_METHODS = [
    "sam2",
]

_SAM2_ZOO_MODEL_NAME = "segment-anything-2-hiera-tiny-video-torch"
_SAM2_LOCAL_TYPE = (
    f"{SegmentAnything2VideoModel.__module__}."
    f"{SegmentAnything2VideoModel.__name__}"
)
_SAM2_LOCAL_MODEL_CACHE: dict[str, Any] = {}


def load_local_sam2(media_mode: str):
    if media_mode in _SAM2_LOCAL_MODEL_CACHE:
        return _SAM2_LOCAL_MODEL_CACHE[media_mode]

    foz.ensure_zoo_model_requirements(
        _SAM2_ZOO_MODEL_NAME, error_level=None, log_success=False
    )
    zoo_model, model_path = foz.download_zoo_model(_SAM2_ZOO_MODEL_NAME)

    config_dict = deepcopy(zoo_model.default_deployment_config_dict)
    config_dict["type"] = _SAM2_LOCAL_TYPE
    config_dict.setdefault("config", {})
    config_dict["config"]["media_mode"] = media_mode

    model = fom.load_model(config_dict, model_path=model_path)
    _SAM2_LOCAL_MODEL_CACHE[media_mode] = model

    return model


def get_frame_field_name(field_name: str, media_mode: str) -> str:
    if media_mode == "video":
        if field_name.startswith("frames."):
            return field_name[len("frames.") :]
    return field_name


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

    run_view = (
        view.sort_by(sort_field)
        if (sort_field and view.has_field(sort_field))
        else view
    )
    run_view.apply_model(
        model,
        # label_field is applied directly to the frame field. hence we need the frame-level field name.
        label_field=get_frame_field_name(output_annotation_field, media_mode),
        prompt_field=input_annotation_field,
        batch_size=int(2 ** np.ceil(np.log2(len(run_view)))),  # type: ignore[arg-type]
        progress=progress,
        skip_failures=False,
    )

    return {}
