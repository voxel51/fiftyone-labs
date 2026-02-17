import logging
from typing import Any

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F
import fiftyone.operators.types as types

from .exemplars import SUPPORTED_SELECTION_METHODS

logger = logging.getLogger(__name__)


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
