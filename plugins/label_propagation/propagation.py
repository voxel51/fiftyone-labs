import numpy as np
import logging
from typing import Tuple, Union, Optional, Any, List

import fiftyone as fo
import fiftyone.zoo as foz


logger = logging.getLogger(__name__)


SUPPORTED_PROPAGATION_METHODS = [
    "sam2",
]


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
    model = foz.load_zoo_model(
        "segment-anything-2-hiera-tiny-video-torch",
        media_mode="image",
    )

    run_view = (
        view.sort_by(sort_field)
        if (sort_field and view.has_field(sort_field))
        else view
    )
    run_view.apply_model(
        model,
        label_field=output_annotation_field,
        prompt_field=input_annotation_field,
        batch_size=int(2**np.ceil(np.log2(len(run_view)))),  # type: ignore[arg-type]
        progress=progress,
        skip_failures=False,
    )

    return {}
