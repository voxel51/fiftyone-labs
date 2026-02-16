"""Video Exemplar Frames plugin.

Extract exemplar frames from a video dataset and propagate annotations.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

import os
import logging
from typing import Any, Dict, List, Optional

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F
import fiftyone.operators.types as types

from .sam2 import propagate_annotations_sam2
from .exemplars import extract_exemplar_frames, SUPPORTED_SELECTION_METHODS


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


class LabelPropagationPanel(foo.Panel):
    """Interactive panel for label propagation with exemplar frames."""

    @property
    def config(self) -> foo.PanelConfig:
        return foo.PanelConfig(
            name="label_propagation",
            label="Label Propagation",
        )
    
    def on_load(self, ctx: Any) -> None:
        ctx.panel.state.base_view = []
        ctx.panel.state.exemplar_frame_field = None
        ctx.panel.state.sort_field = None
        ctx.panel.state.exemplar_field_exists_and_is_populated = False
        ctx.panel.state.exemplars = {}
        ctx.panel.state.selected_exemplar = None
        ctx.panel.state.input_annotation_field = None
        ctx.panel.state.output_annotation_field = None
    
    def register_base_view(self, ctx: Any) -> None:
        """
        - Persist the base view to ctx.panel.base_view
          in a serializable format
        """
        ctx.panel.state.base_view = list(ctx.view.values("id"))
    
    def get_base_view(self, ctx: Any) -> fo.DatasetView:
        if hasattr(ctx.panel.state, "base_view") and ctx.panel.state.base_view:
            return ctx.dataset.select(ctx.panel.state.base_view)
        return ctx.view
    
    def _handle_sort_field_change(self, ctx: Any) -> None:
        """
        - Persist the sort field to ctx.panel.state
        - Sort the view by the sort field
        """
        if "sort_field" in ctx.params:
            ctx.panel.state.sort_field = ctx.params["sort_field"]
        
        sort_field = getattr(ctx.panel.state, "sort_field", None)
        if sort_field:
            ctx.ops.set_view(ctx.view.sort_by(sort_field))
    
    def _handle_exemplar_frame_field_change(self, ctx: Any) -> None:
        """
        - Persist the exemplar frame field to ctx.panel.state
        - Check if the exemplar field exists and is fully populated
        """
        if "exemplar_frame_field" in ctx.params:
            ctx.panel.state.exemplar_frame_field = ctx.params["exemplar_frame_field"]
        
        self._check_exemplar_field_populated(ctx)
        self._discover_exemplars(ctx)
    
    def _check_exemplar_field_populated(
        self, ctx: Any
    ) -> None:
        """
        - Check if exemplar field exists and is fully populated
        - Persist the result to ctx.panel.state.exemplar_field_exists_and_is_populated
        """
        exemplar_frame_field = getattr(ctx.panel.state, "exemplar_frame_field", None)
        if not exemplar_frame_field or exemplar_frame_field not in ctx.dataset.get_field_schema():
            ctx.panel.state.exemplar_field_exists_and_is_populated = False
            return

        view = ctx.view
        samples_with_exemplar = view.exists(exemplar_frame_field)
        count_with_exemplar = len(samples_with_exemplar)
        count_total = len(view)

        ctx.panel.state.exemplar_field_exists_and_is_populated = count_with_exemplar == count_total

    def _discover_exemplars(
        self, ctx: Any
    ) -> None:
        """
        - Create a dict with exemplar 'id's as keys
          and a list of their children's 'id's as values.
          There may be an overlap between the
          samples assigned to different exemplars.
        - Persist the result to ctx.panel.state.exemplars
        - Update the view to only include exemplar samples
        """
        view = self.get_base_view(ctx)
        exemplar_frame_field = getattr(ctx.panel.state, "exemplar_frame_field", None)
        if not exemplar_frame_field:
            return

        # Find all samples where is_exemplar is True
        exemplar_samples = view.match(
            F(f"{exemplar_frame_field}.is_exemplar") == True
        )
        # Get all samples with exemplar_assignment containing the exemplar_id
        for exemplar in exemplar_samples:
            exemplar_id = exemplar.id
            samples_with_exemplar = view.match(
                F(f"{exemplar_frame_field}.exemplar_assignment").contains(
                    exemplar_id
                )
            )
            children_ids = list(samples_with_exemplar.values("id"))
            children_ids.append(exemplar_id)
            ctx.panel.state.exemplars[exemplar_id] = children_ids
        
        # Update the view to only include exemplar
        # with ids in ctx.panel.state.exemplars
        ctx.ops.set_view(view.select(list(ctx.panel.state.exemplars.keys())))
        return
    
    def _handle_input_annotation_field_change(self, ctx: Any) -> None:
        """
        - Persist the input annotation field to ctx.panel.state
        """
        if "input_annotation_field" in ctx.params:
            ctx.panel.state.input_annotation_field = ctx.params["input_annotation_field"]

    def _handle_output_annotation_field_change(self, ctx: Any) -> None:
        """
        - Persist the output annotation field to ctx.panel.state
        """
        if "output_annotation_field" in ctx.params:
            ctx.panel.state.output_annotation_field = ctx.params["output_annotation_field"]

    def _run_assign_exemplar_frames(self, ctx: Any) -> None:
        """Execute AssignExemplarFrames operator."""
        op_ctx = {
            "dataset": ctx.dataset,
            "view": ctx.view,
            "params": {
                "exemplar_frame_field": getattr(ctx.panel.state, "exemplar_frame_field", None),
                "sort_field": getattr(ctx.panel.state, "sort_field", None),
                "method": getattr(ctx.panel.state, "method", None),
            },
        }
        result = foo.execute_operator(
            "@51labs/label_propagation/assign_exemplar_frames",
            op_ctx,
        )

        if result and hasattr(result, "result"):
            message = result.result.get("message", "assign_exemplar_frames operator executed")  # type: ignore[attr-defined]
            ctx.ops.notify(message, variant="success")
        else:
            ctx.ops.notify("Failed to run assign_exemplar_frames operator", variant="error")

        self._handle_exemplar_frame_field_change(ctx)
    
    def _handle_exemplar_selection(self, ctx: Any) -> None:
        """
        - Persist the selected exemplar to ctx.panel.state
        - Open the propagation view for the selected exemplar
        """
        if "selected_exemplar" in ctx.params:
            ctx.panel.state.selected_exemplar = ctx.params["selected_exemplar"]
        
        selected_exemplar = getattr(ctx.panel.state, "selected_exemplar", None)
        if selected_exemplar:
            try:
                propagation_view = self._create_propagation_view(
                    ctx, selected_exemplar
                )
                ctx.ops.set_view(propagation_view)
                assert len(propagation_view) == len(ctx.view)
                # TODO(neeraja): why does the above not work?
                ctx.ops.notify(
                    f"Opened propagation view for sample {selected_exemplar}",
                    variant="info",
                )
            except Exception as e:
                error_msg = f"Failed to open propagation view: {str(e)}"
                logger.error(error_msg, exc_info=True)
                ctx.ops.notify(error_msg, variant="error")

    def _create_propagation_view(
        self,
        ctx: Any,
        sample_id: str,
    ) -> fo.DatasetView:
        """Create a propagation view for the given exemplar.

        Starts from the base view (stored on panel load) to preserve user filters
        while replacing any existing exemplar filters.
        """
        discovered_exemplars = getattr(ctx.panel.state, "exemplars", {})

        if sample_id not in discovered_exemplars:
            exemplar_frame_field = getattr(ctx.panel.state, "exemplar_frame_field", None)
            if not exemplar_frame_field:
                raise RuntimeError(f"Exemplar frame field {exemplar_frame_field} not set")
            sample = ctx.dataset[sample_id]
            sample_exemplar_field = sample.get_field(exemplar_frame_field)
            if not sample_exemplar_field:
                raise RuntimeError(f"Exemplar frame field {exemplar_frame_field} not set for {sample_id}")
            exemplar_ids = sample_exemplar_field["exemplar_assignment"]
        else:
            exemplar_ids = [sample_id]
        
        exemplar_children_ids = []
        for exemplar_id in exemplar_ids:
            exemplar_children_ids.extend(discovered_exemplars[exemplar_id])

        # Create a view with all children sample IDs
        base_view = self.get_base_view(ctx)
        propagation_view = base_view.select(exemplar_children_ids)

        # Sort by sort_field if it exists
        sort_field = getattr(ctx.panel.state, "sort_field", None)
        if sort_field:
            propagation_view = propagation_view.sort_by(sort_field)

        return propagation_view
    
    def _run_propagate_labels(self, ctx: Any) -> None:
        """Execute PropagateLabels operator."""
        op_ctx = {
            "dataset": ctx.dataset,
            "view": ctx.view,
            "params": {
                "input_annotation_field": getattr(ctx.panel.state, "input_annotation_field", None),
                "output_annotation_field": getattr(ctx.panel.state, "output_annotation_field", None),
                "sort_field": getattr(ctx.panel.state, "sort_field", None),
            },
        }
        result = foo.execute_operator(
            "@51labs/label_propagation/propagate_labels",
            op_ctx,
        )

        if result and hasattr(result, "result"):
            message = result.result.get("message", "propagate_labels operator executed")  # type: ignore[attr-defined]
            ctx.ops.notify(message, variant="success")
        else:
            ctx.ops.notify("Failed to run propagate_labels operator", variant="error")
    
    def render(self, ctx: Any) -> types.Property:
        """Render the panel UI."""
        panel = types.Object()

        panel.md("### Label Propagation Across Frames", name="title")

        # Configuration inputs (always at top)
        panel.md("#### Configuration", name="panel_config_header")
        schema = ctx.dataset.get_field_schema()
        field_choices = [types.Choice(label=f, value=f) for f in schema.keys()]
        panel.str(
            "sort_field",
            label="Sort Field",
            default=getattr(ctx.panel.state, "sort_field", None),
            view=types.AutocompleteView(choices=field_choices)
            if field_choices
            else None,
            description="Field to sort samples by",
            on_change=self._handle_sort_field_change,
        )
        panel.str(
            "exemplar_frame_field",
            label="Exemplar Frame Field",
            default=getattr(ctx.panel.state, "exemplar_frame_field", None),
            description="Field name for storing exemplar frame information",
            on_change=self._handle_exemplar_frame_field_change,
        )

        # Assign Exemplar Frames section
        field_exists_and_is_populated = getattr(ctx.panel.state, "exemplar_field_exists_and_is_populated", False)
        if field_exists_and_is_populated:
            panel.md("#### Rerun Exemplar Frame Selection (Optional)", name="panel_exemplar_selection_header_rerun")
        else:
            panel.md("#### Exemplar Frame Selection", name="panel_exemplar_selection_header")
        method_dropdown = types.DropdownView()
        for choice in SUPPORTED_SELECTION_METHODS:
            method_dropdown.add_choice(choice, label=choice)

        panel.str(
            "method",
            label="Method",
            view=method_dropdown,
            default=SUPPORTED_SELECTION_METHODS[0],
            description="Exemplar extraction method",
        )
        panel.btn(
            "run_assign_exemplar_frames",
            label="Run Assign Exemplar Frames",
            on_click=self._run_assign_exemplar_frames,
            variant="contained",
        )

        panel.md("#### Open Propagation View", name="panel_propagation_view_header")
        if field_exists_and_is_populated:
            if not hasattr(ctx.panel.state, "exemplars"):
                self._discover_exemplars(ctx)
            exemplars = getattr(ctx.panel.state, "exemplars", {})

            if exemplars:
                exemplar_dropdown = types.DropdownView()
                for exemplar_id, children_ids in exemplars.items():
                    label = f"Exemplar {exemplar_id} [{len(children_ids)} samples]"
                    exemplar_dropdown.add_choice(exemplar_id, label=label)

                panel.str(
                    "selected_exemplar",
                    label="Selected Exemplar",
                    view=exemplar_dropdown,
                    default=None,
                    on_change=self._handle_exemplar_selection,
                )

            else:
                panel.md(
                    "⚠️ No exemplars found. Run Assign Exemplar Frames first.",
                    name="no_exemplars_warning"
                )
        else:
            panel.md(
                "⚠️ Exemplar field not found or not fully populated. Run Assign Exemplar Frames first.",
                name="field_not_populated_warning"
            )

        # Propagation section
        panel.md("#### Propagation")
        panel.str(
            "input_annotation_field",
            label="Input Annotation Field",
            default=getattr(ctx.panel.state, "input_annotation_field"),
            view=types.AutocompleteView(choices=field_choices)
            if field_choices
            else None,
            required=True,
            description="Field containing annotations to propagate from",
            on_change=self._handle_input_annotation_field_change,
        )
        input_annotation_field = getattr(ctx.panel.state, "input_annotation_field", None)
        if input_annotation_field:
            default_output_annotation_field = input_annotation_field + "_propagated"
        else:
            default_output_annotation_field = getattr(ctx.panel.state, "output_annotation_field", None)
            
        panel.str(
            "output_annotation_field",
            label="Output Annotation Field",
            default=default_output_annotation_field,
            description="Field to store propagated annotations (default: {input_field}_propagated)",
            required=False,
            on_change=self._handle_output_annotation_field_change,
        )

        panel.btn(
            "run_propagate_labels",
            label="Run Propagation",
            on_click=self._run_propagate_labels,
            variant="contained",
        )

        return types.Property(panel)


def register(p):
    p.register(AssignExemplarFrames)
    p.register(PropagateLabels)
    p.register(LabelPropagationPanel)
