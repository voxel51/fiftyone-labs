"""Video Exemplar Frames plugin.

Extract exemplar frames from a video dataset and propagate annotations.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

import logging

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.utils.labels as foul

from .utils import get_frame_schema, auth_context
from .exemplars import (
    SUPPORTED_TEMPORAL_SEGMENTATION_METHODS,
    SUPPORTED_EXEMPLAR_SCORING_METHODS,
    extract_temporal_segments,
    select_exemplars,
)
from .propagation import (
    SAM2_PROPAGATION_METHODS,
    propagate_annotations_sam2,
    propagate_annotations_sam3,
)
from .propagation_cv2 import CV2_PROPAGATION_METHODS, propagate_annotations_cv2
from .propagation_grabcut import GRABCUT_PROPAGATION_METHODS, propagate_annotations_grabcut
from .propagation_densecrf import DENSECRF_PROPAGATION_METHODS, propagate_annotations_densecrf

SUPPORTED_PROPAGATION_METHODS = (
    ["sam2"] + SAM2_PROPAGATION_METHODS + ["sam3"]
    + CV2_PROPAGATION_METHODS
    + GRABCUT_PROPAGATION_METHODS
    + DENSECRF_PROPAGATION_METHODS
)
from .panel import LabelPropagationPanel


logger = logging.getLogger(__name__)


class TemporalSegmentation(foo.Operator):
    version = "1.0.0"

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="temporal_segmentation",
            label="Temporal Segmentation",
            description="Label chunks of frames into temporal segments",
            light_icon="/assets/labs_icon_light.svg",
            dark_icon="/assets/labs_icon_dark.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()
        inputs.view_target(ctx)

        method_dropdown = types.Dropdown()
        for choice in SUPPORTED_TEMPORAL_SEGMENTATION_METHODS:
            method_dropdown.add_choice(choice, label=choice)

        inputs.enum(
            "temporal_segmentation_method",
            method_dropdown.values(),
            default=SUPPORTED_TEMPORAL_SEGMENTATION_METHODS[0],
            label="Segmentation Method",
            view=method_dropdown,
            required=True,
        )

        inputs.str(
            "temporal_segments_field",
            label="Temporal Segments Field",
            default=None,
            required=True,
        )

        schema = get_frame_schema(ctx.target_view())
        field_choices = [types.Choice(label=f, value=f) for f in schema.keys()]
        inputs.str(
            "sort_field",
            label="Field to Sort Samples by",
            default="frame_number",
            view=types.AutocompleteView(choices=field_choices)
            if field_choices
            else None,
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx) -> dict:
        temporal_segmentation_method = ctx.params.get(
            "temporal_segmentation_method"
        )
        temporal_segments_field = ctx.params.get("temporal_segments_field")
        sort_field = ctx.params.get("sort_field", None)

        dataset = ctx.dataset
        schema = dataset.get_field_schema()
        if temporal_segments_field in schema:
            ft = type(schema[temporal_segments_field]).__name__
            if ft != "EmbeddedDocumentField":
                logger.warning(
                    f"Found existing field '{temporal_segments_field}' with type '{ft}'. This will be overwritten."
                )
                dataset.delete_sample_field(
                    temporal_segments_field, error_level=2
                )

        if temporal_segments_field not in dataset.get_field_schema():
            if dataset.media_type == "video":
                dataset.add_sample_field(
                    temporal_segments_field,
                    fo.EmbeddedDocumentField,
                    embedded_doc_type=fo.TemporalDetections,
                )
                dataset.add_sample_field(
                    f"{temporal_segments_field}.detections.exemplar_score",
                    fo.FloatField,
                )
            else:
                dataset.add_sample_field(
                    temporal_segments_field,
                    fo.EmbeddedDocumentField,
                    embedded_doc_type=fo.Classifications,
                )
                dataset.add_sample_field(
                    f"{temporal_segments_field}.classifications.exemplar_score",
                    fo.FloatField,
                )

        try:
            extract_temporal_segments(
                view=ctx.target_view(),
                method=temporal_segmentation_method,
                temporal_segments_field=temporal_segments_field,
                sort_field=sort_field,
            )
        except (RuntimeError, ValueError) as e:
            error_msg = str(e)
            logger.error(error_msg)
            ctx.ops.notify(
                error_msg,
                variant="error",
            )
            return {
                "message": error_msg,
                "samples_processed": 0,
            }

        return {
            "message": f"Temporal segments stored in '{temporal_segments_field}'",
            "samples_processed": len(ctx.target_view()),
        }


class SelectExemplars(foo.Operator):
    version = "1.0.0"

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="select_exemplars",
            label="Select Exemplars",
            description="Set exemplar scores on temporal segment classifications",
            light_icon="/assets/labs_icon_light.svg",
            dark_icon="/assets/labs_icon_dark.svg",
            dynamic=True,
        )

    def validate_input(self, ctx) -> bool:
        temporal_segments_field = ctx.params.get("temporal_segments_field")
        schema = get_frame_schema(ctx.target_view())
        if temporal_segments_field in schema:
            ft = type(schema[temporal_segments_field]).__name__
            if ft != "EmbeddedDocumentField":
                logger.warning(
                    f"'{temporal_segments_field}' field exists but with type '{ft}'. temporal_segments_field should be of type fo.EmbeddedDocumentField."
                )
                return False
        else:
            logger.warning(
                f"'{temporal_segments_field}' field not found in the dataset. temporal_segments_field should exist and be of type fo.EmbeddedDocumentField."
            )
            return False
        return True

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()
        inputs.view_target(ctx)

        inputs.str(
            "temporal_segments_field",
            label="Temporal Segments Field",
            default=None,
            required=True,
        )

        method_dropdown = types.Dropdown()
        for choice in SUPPORTED_EXEMPLAR_SCORING_METHODS:
            method_dropdown.add_choice(choice, label=choice)

        inputs.enum(
            "exemplar_scoring_method",
            method_dropdown.values(),
            default=SUPPORTED_EXEMPLAR_SCORING_METHODS[0],
            label="Exemplar Selection Method",
            view=method_dropdown,
            required=True,
        )

        schema = get_frame_schema(ctx.target_view())
        field_choices = [types.Choice(label=f, value=f) for f in schema.keys()]
        inputs.str(
            "sort_field",
            label="Field to Sort Samples by",
            default="frame_number",
            view=types.AutocompleteView(choices=field_choices)
            if field_choices
            else None,
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx) -> dict:
        if not self.validate_input(ctx):
            return {
                "message": "Validation failed",
                "samples_processed": 0,
            }

        temporal_segments_field = ctx.params.get("temporal_segments_field")
        exemplar_scoring_method = ctx.params.get("exemplar_scoring_method")
        sort_field = ctx.params.get("sort_field", None)

        select_exemplars(
            view=ctx.target_view(),
            temporal_segments_field=temporal_segments_field,
            method=exemplar_scoring_method,
            sort_field=sort_field,
        )

        return {
            "message": f"Exemplar scores set in '{temporal_segments_field}'",
            "samples_processed": len(ctx.target_view()),
        }


class PropagateLabels(foo.Operator):
    version = "1.0.0"

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="propagate_labels",
            label="Propagate Labels From Input Field Operator",
            description="Propagate labels from labeled frames to all frames",
            light_icon="/assets/labs_icon_light.svg",
            dark_icon="/assets/labs_icon_dark.svg",
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    def validate_input(self, ctx) -> bool:
        input_annotation_field = ctx.params.get("input_annotation_field", None)
        if input_annotation_field is None:
            logger.warning(
                "Input annotation field is not provided. Please provide a field name to propagate from."
            )
            return False

        output_annotation_field = ctx.params.get(
            "output_annotation_field", input_annotation_field + "_propagated"
        )
        if output_annotation_field == input_annotation_field:
            logger.warning(
                f"Output annotation field '{output_annotation_field}' cannot be the same as "
                f"the input annotation field '{input_annotation_field}'. "
                f"Please choose a different output field name to avoid overwriting the source annotations."
            )
            return False

        schema = get_frame_schema(ctx.target_view())
        if input_annotation_field not in schema:
            logger.warning(
                f"Input annotation field '{input_annotation_field}' not found in the dataset. "
                f"Please ensure the field exists and contains annotations."
            )
            return False

        batch_size = ctx.params.get("batch_size", 32)
        if ctx.dataset.media_type != "video" and batch_size < 2:
            logger.warning(
                f"Batch size '{batch_size}' has to be >= 2 for propagation to work."
            )

        return True

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()
        inputs.view_target(ctx)

        # Get available fields from dataset schema for autocomplete
        schema = get_frame_schema(ctx.target_view())
        field_choices = [types.Choice(label=f, value=f) for f in schema.keys()]

        inputs.str(
            "input_annotation_field",
            label="Annotation Field to Propagate from",
            view=types.AutocompleteView(choices=field_choices)
            if field_choices
            else None,
            required=True,
        )

        inputs.str(
            "output_annotation_field",
            label="Annotation Field to Propagate to",
            description="If not provided, a new field will be created with the name of the input field plus '_propagated'",
            required=False,
        )

        propagation_method_dropdown = types.Dropdown()
        for choice in SUPPORTED_PROPAGATION_METHODS:
            propagation_method_dropdown.add_choice(choice, label=choice)

        inputs.enum(
            "propagation_method",
            propagation_method_dropdown.values(),
            default=SUPPORTED_PROPAGATION_METHODS[0],
            label="Propagation Method",
            view=propagation_method_dropdown,
            required=True,
        )

        inputs.str(
            "sort_field",
            label="Field to Sort Samples by",
            default="frame_number",
            view=types.AutocompleteView(choices=field_choices)
            if field_choices
            else None,
            required=False,
        )

        inputs.int(
            "batch_size",
            label="Batch Size",
            description="Maximum number of media samples to process in one pass. Reduce if you run out of memory.",
            min=1,
            default=32,
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx) -> dict:
        if not self.validate_input(ctx):
            return {
                "message": "Validation failed",
                "samples_processed": 0,
            }

        view = ctx.target_view()
        total_samples = len(view)
        input_annotation_field = ctx.params.get("input_annotation_field")
        output_annotation_field = ctx.params.get(
            "output_annotation_field", None
        )
        if (output_annotation_field is None) or len(
            output_annotation_field
        ) == 0:
            output_annotation_field = f"{input_annotation_field}_propagated"
        propagation_method = ctx.params.get("propagation_method")
        sort_field = ctx.params.get("sort_field", None)
        batch_size = ctx.params.get("batch_size", 32)

        try:
            if propagation_method in SAM2_PROPAGATION_METHODS or propagation_method == "sam2":
                _ = propagate_annotations_sam2(
                    view=view,
                    input_annotation_field=input_annotation_field,
                    output_annotation_field=output_annotation_field,
                    sort_field=sort_field,
                    batch_size=batch_size,
                    progress=True,
                    propagation_method=propagation_method,
                )
            elif propagation_method == "sam3":
                _ = propagate_annotations_sam3(
                    view=view,
                    input_annotation_field=input_annotation_field,
                    output_annotation_field=output_annotation_field,
                    sort_field=sort_field,
                    batch_size=batch_size,
                    progress=True,
                )
            elif propagation_method in CV2_PROPAGATION_METHODS:
                _ = propagate_annotations_cv2(
                    view=view,
                    input_annotation_field=input_annotation_field,
                    output_annotation_field=output_annotation_field,
                    method=propagation_method,
                    sort_field=sort_field,
                    progress=True,
                )
            elif propagation_method in GRABCUT_PROPAGATION_METHODS:
                _ = propagate_annotations_grabcut(
                    view=view,
                    input_annotation_field=input_annotation_field,
                    output_annotation_field=output_annotation_field,
                    sort_field=sort_field,
                    progress=True,
                )
            elif propagation_method in DENSECRF_PROPAGATION_METHODS:
                _ = propagate_annotations_densecrf(
                    view=view,
                    input_annotation_field=input_annotation_field,
                    output_annotation_field=output_annotation_field,
                    sort_field=sort_field,
                    progress=True,
                )
            else:
                raise RuntimeError(
                    f"Unsupported propagation method '{propagation_method}'"
                )
            if view.media_type == "video":
                # instances keyed by (id, label, index); not common for image datasets
                with auth_context():
                    foul.index_to_instance(view, output_annotation_field)

        except (RuntimeError, ValueError) as e:
            error_msg = str(e)
            logger.error(error_msg)
            ctx.ops.notify(
                error_msg,
                variant="error",
            )
            return {
                "message": error_msg,
                "samples_processed": 0,
            }

        if not ctx.delegated:
            ctx.ops.reload_dataset()

        return {
            "message": f"Annotations propagated from {input_annotation_field} to {output_annotation_field}",
            "samples_processed": total_samples,
        }


def register(p):
    p.register(TemporalSegmentation)
    p.register(SelectExemplars)
    p.register(PropagateLabels)
    p.register(LabelPropagationPanel)
