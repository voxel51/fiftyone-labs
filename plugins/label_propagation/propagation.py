import logging
from typing import Tuple, Union, Optional, Any, List

import fiftyone as fo
import fiftyone.zoo as foz


logger = logging.getLogger(__name__)


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

    if view.has_field("tags"):
        all_tags = view.distinct("tags") or []
        sequence_tags = [t for t in all_tags if t != "val"]
        for tag in sequence_tags:
            seq_view = view.match_tags(tag)
            if sort_field and view.has_field(sort_field):
                seq_view = seq_view.sort_by(sort_field)
            seq_view.apply_model(
                model,
                label_field=output_annotation_field,
                prompt_field=input_annotation_field,
                progress=progress,
            )
    else:
        run_view = (
            view.sort_by(sort_field)
            if (sort_field and view.has_field(sort_field))
            else view
        )
        run_view.apply_model(
            model,
            label_field=output_annotation_field,
            prompt_field=input_annotation_field,
            progress=progress,
        )

    return {}
