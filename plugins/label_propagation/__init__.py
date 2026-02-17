"""Video Exemplar Frames plugin.

Extract exemplar frames from a video dataset and propagate annotations.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

import logging

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .sam2 import propagate_annotations_sam2
from .exemplars import extract_exemplar_frames, SUPPORTED_SELECTION_METHODS
from .panel import LabelPropagationPanel


logger = logging.getLogger(__name__)


class AssignExemplarFrames(foo.Operator):
    version = "1.0.0"

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="assign_exemplar_frames",
            label="Assign Exemplar Frames Operator",
            description="Assign exemplar frames to frames of a video",
            light_icon="/assets/labs_icon_light.svg",
            dark_icon="/assets/labs_icon_dark.svg",
            dynamic=True,
            execution_options=foo.ExecutionOptions(
                allow_immediate=False,
                allow_delegation=True,
                default_choice_to_delegated=True,
            ),
        )

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()
        inputs.view_target(ctx)

        method_dropdown = types.Dropdown()
        for choice in SUPPORTED_SELECTION_METHODS:
            method_dropdown.add_choice(choice, label=choice)

        inputs.enum(
            "method",
            method_dropdown.values(),
            default=SUPPORTED_SELECTION_METHODS[0],
            label="Exemplar Extraction Method",
            view=method_dropdown,
            required=True,
        )

        # exemplar frame field
        inputs.str(
            "exemplar_frame_field",
            label="Exemplar Frame Information Field",
            default="exemplar",
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
        method = ctx.params.get("method")
        exemplar_frame_field = ctx.params.get("exemplar_frame_field")
        sort_field = ctx.params.get("sort_field", None)

        # Check if field exists and validate/convert its type
        dataset = ctx.dataset
        if exemplar_frame_field in dataset.get_field_schema():
            logger.debug(
                f"Exemplar frame field exists: {exemplar_frame_field}"
            )
            field_type_name = type(
                dataset.get_field_schema()[exemplar_frame_field]
            ).__name__
            if field_type_name != "EmbeddedDocumentField":
                logger.warning(
                    f"Deleting exemplar frame field of incorrect type: {exemplar_frame_field}"
                )
                dataset.delete_sample_field(
                    exemplar_frame_field, error_level=2
                )

        # Ensure the exemplar field exists and declare nested fields for proper schema support
        # Use DynamicEmbeddedDocument to allow dynamic fields
        if exemplar_frame_field not in dataset.get_field_schema():
            logger.info(f"Adding exemplar frame field: {exemplar_frame_field}")
            dataset.add_sample_field(
                exemplar_frame_field,
                fo.EmbeddedDocumentField,
                embedded_doc_type=fo.DynamicEmbeddedDocument,
            )
            dataset.add_sample_field(
                f"{exemplar_frame_field}.is_exemplar", fo.BooleanField
            )
            dataset.add_sample_field(
                f"{exemplar_frame_field}.exemplar_assignment",
                fo.ListField,
                subfield=fo.ObjectIdField,
            )

        extract_exemplar_frames(
            view=ctx.target_view(),
            method=method,
            exemplar_frame_field=exemplar_frame_field,
            sort_field=sort_field,
        )

        return {
            "message": f"Exemplar frames extracted and stored in field '{exemplar_frame_field}'",
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
            execution_options=foo.ExecutionOptions(
                allow_immediate=False,
                allow_delegation=True,
                default_choice_to_delegated=True,
            ),
        )

    def validate_input(self, ctx) -> bool:
        input_annotation_field = ctx.params.get(
            "input_annotation_field", "human_labels"
        )
        output_annotation_field = ctx.params.get(
            "output_annotation_field", "human_labels_propagated"
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
            default="human_labels",
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
            "output_annotation_field", f"{input_annotation_field}_propagated"
        )
        sort_field = ctx.params.get("sort_field", None)

        try:
            _ = propagate_annotations_sam2(
                view=view,
                input_annotation_field=input_annotation_field,
                output_annotation_field=output_annotation_field,
                sort_field=sort_field,
                progress=True,
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
    p.register(AssignExemplarFrames)
    p.register(PropagateLabels)
    p.register(LabelPropagationPanel)
