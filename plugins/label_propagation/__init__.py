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
from .exemplars import extract_exemplar_frames


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

        method_choices = [
            "heuristic",
            # TODO(neeraja): add PySceneDetect
            # TODO(neeraja): add Embedding-based
        ]
        method_dropdown = types.Dropdown()
        for choice in method_choices:
            method_dropdown.add_choice(choice, label=choice)

        inputs.enum(
            "method",
            method_dropdown.values(),
            default="heuristic",
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
            required=True,
        )

        return types.Property(inputs)

    def execute(self, ctx) -> dict:
        method = ctx.params.get("method")
        exemplar_frame_field = ctx.params.get("exemplar_frame_field")
        sort_field = ctx.params.get("sort_field")

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
            required=True,
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
        sort_field = ctx.params.get("sort_field")

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

    def _get_base_view(self, ctx: Any) -> fo.DatasetView:
        """Get the base view to use for propagation views.
        
        Always starts from the dataset to avoid stacking exemplar filters.
        Users should apply their filters to the dataset before using the panel.
        """
        return ctx.dataset

    def _get_exemplar_frame_field(self, ctx: Any) -> str:
        """Get exemplar_frame_field from panel state or default."""
        return getattr(ctx.panel.state, "exemplar_frame_field", "exemplar")

    def _get_sort_field(self, ctx: Any) -> str:
        """Get sort_field from panel state or default."""
        return getattr(ctx.panel.state, "sort_field", "frame_number")

    def _check_exemplar_field_populated(
        self, ctx: Any, exemplar_frame_field: str
    ) -> tuple[bool, bool]:
        """
        Check if exemplar field exists and is fully populated.
        Returns: (field_exists, is_fully_populated)
        """
        schema = ctx.dataset.get_field_schema()
        if exemplar_frame_field not in schema:
            return False, False

        # Check if field is fully populated in current view
        view = ctx.view
        total_samples = len(view)
        if total_samples == 0:
            return True, False

        # Count samples with exemplar data
        samples_with_exemplar = view.exists(exemplar_frame_field)
        count_with_exemplar = len(samples_with_exemplar)

        # Field is fully populated if all samples have exemplar data
        is_fully_populated = count_with_exemplar == total_samples
        return True, is_fully_populated

    def _discover_exemplars(
        self, ctx: Any, exemplar_frame_field: str
    ) -> List[Dict[str, Any]]:
        """
        Discover all exemplars from the dataset.
        Returns list of dicts with 'id' and 'sample_count'.
        """
        view = ctx.view
        exemplars = []

        # Find all samples where is_exemplar is True
        exemplar_samples = view.match(
            F(f"{exemplar_frame_field}.is_exemplar") == True
        )

        for sample in exemplar_samples:
            exemplar_id = sample.id

            # Count samples in propagation view (where exemplar_id is in exemplar_assignment)
            # Include the exemplar sample itself
            samples_with_exemplar = view.match(
                F(f"{exemplar_frame_field}.exemplar_assignment").contains(exemplar_id)
            )
            # Get all sample IDs from both views and combine
            sample_ids = list(samples_with_exemplar.values("id"))
            # Add exemplar_id if not already present
            if exemplar_id not in sample_ids:
                sample_ids.append(exemplar_id)
            propagation_view = view.select(sample_ids)
            sample_count = len(propagation_view)

            exemplars.append({"id": exemplar_id, "sample_count": sample_count})

        return exemplars

    def _create_propagation_view(
        self, ctx: Any, exemplar_id: str, exemplar_frame_field: str, sort_field: str
    ) -> fo.DatasetView:
        """Create a propagation view for the given exemplar.
        
        Starts from the base view (stored on panel load) to preserve user filters
        while replacing any existing exemplar filters.
        """
        # Start from base view to preserve user filters but avoid stacking exemplar filters
        base_view = self._get_base_view(ctx)
        
        # Filter to samples where exemplar_id is in exemplar_assignment
        samples_with_exemplar = base_view.match(
            F(f"{exemplar_frame_field}.exemplar_assignment").contains(exemplar_id)
        )
        
        # Get all sample IDs from the filtered view
        sample_ids = list(samples_with_exemplar.values("id"))
        # Add exemplar_id if not already present (it might be if exemplar_assignment contains itself)
        if exemplar_id not in sample_ids:
            sample_ids.append(exemplar_id)
        
        # Create view with all sample IDs
        propagation_view = base_view.select(sample_ids)

        # Sort by sort_field if it exists
        schema = ctx.dataset.get_field_schema()
        if sort_field in schema:
            try:
                propagation_view = propagation_view.sort_by(sort_field)
            except Exception as e:
                logger.warning(f"Could not sort by {sort_field}: {e}")

        return propagation_view

    def _run_assign_exemplar_frames(self, ctx: Any) -> None:
        """Execute AssignExemplarFrames operator."""
        exemplar_frame_field = self._get_exemplar_frame_field(ctx)
        sort_field = self._get_sort_field(ctx)
        method = getattr(ctx.panel.state, "method", "heuristic")

        # Create operator context
        op_ctx = {
            "dataset": ctx.dataset,
            "view": ctx.view,
            "params": {
                "exemplar_frame_field": exemplar_frame_field,
                "sort_field": sort_field,
                "method": method,
            },
        }

        # Execute operator
        result = foo.execute_operator(
            "@51labs/label_propagation/assign_exemplar_frames",
            op_ctx,
        )

        if result and hasattr(result, "result"):
            message = result.result.get("message", "Exemplar frames assigned")  # type: ignore[attr-defined]
            ctx.ops.notify(message, variant="success")
        else:
            ctx.ops.notify("Exemplar frames assigned", variant="success")

        # Refresh exemplars after execution
        self._refresh_exemplars(ctx)


    def _run_propagate_labels(self, ctx: Any) -> None:
        """Execute PropagateLabels operator."""
        input_annotation_field = getattr(
            ctx.panel.state, "input_annotation_field", "human_labels"
        )
        output_annotation_field = getattr(
            ctx.panel.state, "output_annotation_field", None
        )
        sort_field = self._get_sort_field(ctx)

        if not output_annotation_field:
            output_annotation_field = f"{input_annotation_field}_propagated"

        try:
            # Create operator context
            op_ctx = {
                "dataset": ctx.dataset,
                "view": ctx.view,
                "params": {
                    "input_annotation_field": input_annotation_field,
                    "output_annotation_field": output_annotation_field,
                    "sort_field": sort_field,
                },
            }

            # Execute operator
            result = foo.execute_operator(
                "@51labs/label_propagation/propagate_labels",
                op_ctx,
            )

            if result and hasattr(result, "result"):
                message = result.result.get("message", "Labels propagated")  # type: ignore[attr-defined]
                ctx.ops.notify(message, variant="success")
            else:
                ctx.ops.notify("Labels propagated", variant="success")

        except Exception as e:
            error_msg = f"Failed to propagate labels: {str(e)}"
            logger.error(error_msg, exc_info=True)
            ctx.ops.notify(error_msg, variant="error")

    def _refresh_exemplars(self, ctx: Any) -> None:
        """Refresh exemplar list from dataset."""
        exemplar_frame_field = self._get_exemplar_frame_field(ctx)
        field_exists, is_populated = self._check_exemplar_field_populated(
            ctx, exemplar_frame_field
        )

        if field_exists and is_populated:
            exemplars = self._discover_exemplars(ctx, exemplar_frame_field)
            ctx.panel.state.exemplars = exemplars
        else:
            ctx.panel.state.exemplars = []

    def _on_exemplar_selected(self, ctx: Any) -> None:
        """Handle exemplar selection change - auto-open propagation view."""
        # The value comes from on_change callback in ctx.params["value"]
        selected_exemplar = ctx.params.get("value")
        if not selected_exemplar:
            # Fallback: check if it's in the field name directly
            selected_exemplar = ctx.params.get("selected_exemplar")
        
        if selected_exemplar:
            ctx.panel.state.selected_exemplar = selected_exemplar
            self._open_propagation_view(ctx)
        else:
            logger.warning(f"No exemplar value found in params: {ctx.params}")

    def _open_propagation_view(self, ctx: Any) -> None:
        """Open propagation view for selected exemplar."""
        selected_exemplar = getattr(ctx.panel.state, "selected_exemplar", None)
        if not selected_exemplar:
            ctx.ops.notify("No exemplar selected", variant="warning")
            return

        exemplar_frame_field = self._get_exemplar_frame_field(ctx)
        sort_field = self._get_sort_field(ctx)

        try:
            propagation_view = self._create_propagation_view(
                ctx, selected_exemplar, exemplar_frame_field, sort_field
            )
            ctx.ops.set_view(propagation_view)
            ctx.ops.notify(
                f"Opened propagation view for exemplar {selected_exemplar[:8]}...",
                variant="info",
            )
        except Exception as e:
            error_msg = f"Failed to open propagation view: {str(e)}"
            logger.error(error_msg, exc_info=True)
            ctx.ops.notify(error_msg, variant="error")

    def _handle_sample_selection(self, ctx: Any) -> None:
        """Handle sample selection to auto-open propagation view."""
        if not ctx.current_sample:
            return

        exemplar_frame_field = self._get_exemplar_frame_field(ctx)
        sample = ctx.dataset[ctx.current_sample]

        # Check if sample has exemplar data
        if exemplar_frame_field not in sample.field_names:
            return

        exemplar_data = sample.get_field(exemplar_frame_field)
        if not exemplar_data:
            return

        # Determine selected exemplar
        if getattr(exemplar_data, "is_exemplar", False):
            selected_exemplar = sample.id
        else:
            exemplar_assignment = getattr(exemplar_data, "exemplar_assignment", [])
            if exemplar_assignment:
                selected_exemplar = exemplar_assignment[0]
            else:
                return

        # Update panel state and open view
        ctx.panel.state.selected_exemplar = selected_exemplar
        self._open_propagation_view(ctx)

    def render(self, ctx: Any) -> types.Property:
        """Render the panel UI."""
        panel = types.Object()

        # Get configuration values
        exemplar_frame_field = self._get_exemplar_frame_field(ctx)
        sort_field = self._get_sort_field(ctx)

        # Update state from params if present
        if "exemplar_frame_field" in ctx.params:
            exemplar_frame_field = ctx.params["exemplar_frame_field"]
            ctx.panel.state.exemplar_frame_field = exemplar_frame_field
        if "sort_field" in ctx.params:
            sort_field = ctx.params["sort_field"]
            ctx.panel.state.sort_field = sort_field

        # Get schema once for reuse
        schema = ctx.dataset.get_field_schema()
        field_choices = [types.Choice(label=f, value=f) for f in schema.keys()]

        # Configuration inputs (always at top)
        panel.md("---\n**Configuration:**")
        panel.str(
            "exemplar_frame_field",
            label="Exemplar Frame Field",
            default=exemplar_frame_field,
            description="Field name for storing exemplar frame information",
        )
        panel.str(
            "sort_field",
            label="Sort Field",
            default=sort_field,
            view=types.AutocompleteView(choices=field_choices)
            if field_choices
            else None,
            description="Field to sort samples by",
        )

        # Check exemplar field status
        field_exists, is_populated = self._check_exemplar_field_populated(
            ctx, exemplar_frame_field
        )

        # Assign Exemplar Frames section
        panel.md("---\n**Assign Exemplar Frames:**")
        method_choices = ["heuristic"]
        method_dropdown = types.DropdownView()
        for choice in method_choices:
            method_dropdown.add_choice(choice, label=choice)

        current_method = getattr(ctx.panel.state, "method", "heuristic")
        panel.str(
            "method",
            label="Method",
            view=method_dropdown,
            default=current_method,
            description="Exemplar extraction method",
        )
        if "method" in ctx.params:
            ctx.panel.state.method = ctx.params["method"]

        panel.btn(
            "run_assign_exemplar_frames",
            label="Run Assign Exemplar Frames",
            on_click=self._run_assign_exemplar_frames,
            variant="contained",
        )

        # Exemplars section
        panel.md("---\n**Exemplars:**")
        if field_exists and is_populated:
            if not hasattr(ctx.panel.state, "exemplars"):
                self._refresh_exemplars(ctx)
            exemplars = getattr(ctx.panel.state, "exemplars", [])

            if exemplars:
                exemplar_dropdown = types.DropdownView()
                for exemplar in exemplars:
                    exemplar_id = exemplar["id"]
                    sample_count = exemplar["sample_count"]
                    label = f"Exemplar {exemplar_id[:8]}... ({sample_count} samples)"
                    exemplar_dropdown.add_choice(exemplar_id, label=label)

                selected_exemplar = getattr(ctx.panel.state, "selected_exemplar", None)
                panel.str(
                    "selected_exemplar",
                    label="Selected Exemplar",
                    view=exemplar_dropdown,
                    default=selected_exemplar if selected_exemplar else None,
                    on_change=self._on_exemplar_selected,
                )

                panel.btn(
                    "refresh_exemplars",
                    label="Refresh Exemplars",
                    on_click=self._refresh_exemplars,
                    variant="outlined",
                )

                if selected_exemplar:
                    exemplar_info = next(
                        (e for e in exemplars if e["id"] == selected_exemplar), None
                    )
                    if exemplar_info:
                        panel.md(
                            f"**Current:** Exemplar {selected_exemplar[:8]}... "
                            f"({exemplar_info['sample_count']} samples)"
                        )
            else:
                panel.md("⚠️ No exemplars found. Run Assign Exemplar Frames first.")
        else:
            panel.md("⚠️ Exemplar field not found or not fully populated. Run Assign Exemplar Frames first.")

        # Propagation section
        panel.md("---\n**Propagation:**")
        input_annotation_field = getattr(
            ctx.panel.state, "input_annotation_field", "human_labels"
        )
        panel.str(
            "input_annotation_field",
            label="Input Annotation Field",
            default=input_annotation_field,
            view=types.AutocompleteView(choices=field_choices)
            if field_choices
            else None,
            description="Field containing annotations to propagate from",
        )
        if "input_annotation_field" in ctx.params:
            ctx.panel.state.input_annotation_field = ctx.params["input_annotation_field"]

        output_annotation_field = getattr(
            ctx.panel.state, "output_annotation_field", None
        )
        panel.str(
            "output_annotation_field",
            label="Output Annotation Field",
            default=output_annotation_field,
            description="Field to store propagated annotations (default: {input_field}_propagated)",
            required=False,
        )
        if "output_annotation_field" in ctx.params:
            ctx.panel.state.output_annotation_field = ctx.params.get(
                "output_annotation_field", None
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