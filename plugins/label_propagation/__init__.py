"""Video Exemplar Frames plugin.

Extract exemplar frames from a video dataset and propagate annotations.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

import logging

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .exemplars import (
    SUPPORTED_EXEMPLAR_SELECTION_METHODS,
    SUPPORTED_SELECTION_METHODS,
    extract_temporal_segments,
    select_exemplars,
)
from .propagation import (
    SUPPORTED_PROPAGATION_METHODS,
    propagate_annotations_sam2,
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
            description="Populate temporal segments field with class labels",
            light_icon="/assets/labs_icon_light.svg",
            dark_icon="/assets/labs_icon_dark.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()
        inputs.view_target(ctx)

        method_dropdown = types.Dropdown()
        for choice in SUPPORTED_SELECTION_METHODS:
            method_dropdown.add_choice(choice, label=choice)

        inputs.enum(
            "selection_method",
            method_dropdown.values(),
            default=SUPPORTED_SELECTION_METHODS[0],
            label="Segmentation Method",
            view=method_dropdown,
            required=True,
        )

        inputs.str(
            "temporal_segments_field",
            label="Temporal Segments Field",
            default="temporal_segments",
            required=True,
        )

        schema = ctx.dataset.get_field_schema()
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
        selection_method = ctx.params.get("selection_method")
        temporal_segments_field = ctx.params.get("temporal_segments_field")
        sort_field = ctx.params.get("sort_field", None)

        dataset = ctx.dataset
        schema = dataset.get_field_schema()
        if temporal_segments_field in schema:
            ft = type(schema[temporal_segments_field]).__name__
            if ft != "EmbeddedDocumentField":
                dataset.delete_sample_field(temporal_segments_field, error_level=2)

        if temporal_segments_field not in dataset.get_field_schema():
            dataset.add_sample_field(
                temporal_segments_field,
                fo.EmbeddedDocumentField,
                embedded_doc_type=fo.Classifications,
            )
            dataset.add_sample_field(
                f"{temporal_segments_field}.classifications.exemplar_score",
                fo.FloatField,
            )

        extract_temporal_segments(
            view=ctx.target_view(),
            method=selection_method,
            temporal_segments_field=temporal_segments_field,
            sort_field=sort_field,
        )

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

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()
        inputs.view_target(ctx)

        inputs.str(
            "temporal_segments_field",
            label="Temporal Segments Field",
            default="temporal_segments",
            required=True,
        )

        method_dropdown = types.Dropdown()
        for choice in SUPPORTED_EXEMPLAR_SELECTION_METHODS:
            method_dropdown.add_choice(choice, label=choice)

        inputs.enum(
            "exemplar_selection_method",
            method_dropdown.values(),
            default=SUPPORTED_EXEMPLAR_SELECTION_METHODS[0],
            label="Exemplar Selection Method",
            view=method_dropdown,
            required=True,
        )

        schema = ctx.dataset.get_field_schema()
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
        temporal_segments_field = ctx.params.get("temporal_segments_field")
        exemplar_selection_method = ctx.params.get("exemplar_selection_method")
        sort_field = ctx.params.get("sort_field", None)

        select_exemplars(
            view=ctx.target_view(),
            temporal_segments_field=temporal_segments_field,
            method=exemplar_selection_method,
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

        schema = ctx.dataset.get_field_schema()
        if input_annotation_field not in schema:
            logger.warning(
                f"Input annotation field '{input_annotation_field}' not found in the dataset. "
                f"Please ensure the field exists and contains annotations."
            )
            return False

        return True

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()
        inputs.view_target(ctx)

        # Get available fields from dataset schema for autocomplete
        schema = ctx.dataset.get_field_schema()
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

        try:
            if propagation_method == "sam2":
                _ = propagate_annotations_sam2(
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
        except RuntimeError as e:
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
            "message": f"Annotations propagated from {input_annotation_field} to {output_annotation_field}",
            "samples_processed": total_samples,
        }


def register(p):
    p.register(TemporalSegmentation)
    p.register(SelectExemplars)
    p.register(PropagateLabels)
    p.register(LabelPropagationPanel)
