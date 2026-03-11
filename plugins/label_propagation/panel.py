import logging
from typing import Any

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .exemplars import (
    SUPPORTED_EXEMPLAR_SELECTION_METHODS,
    SUPPORTED_TEMPORAL_SEGMENTATION_METHODS,
)
from .propagation import SUPPORTED_PROPAGATION_METHODS


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
        ctx.panel.state.temporal_segments_field = None
        ctx.panel.state.sort_field = None
        ctx.panel.state.temporal_segments_populated = False
        ctx.panel.state.segments = {}
        ctx.panel.state.selected_segment = None
        ctx.panel.state.input_annotation_field = None
        ctx.panel.state.output_annotation_field = None
        self.register_base_view(ctx)

    def register_base_view(self, ctx: Any) -> None:
        """
        - Persist the base view to ctx.panel.base_view
          in a serializable format
        """
        logger.info(f"Registering base view with {len(ctx.view)} samples")
        ctx.panel.state.base_view = list(ctx.view.values("id"))

    def get_base_view(self, ctx: Any) -> fo.DatasetView:
        if hasattr(ctx.panel.state, "base_view") and ctx.panel.state.base_view:
            return ctx.dataset.select(ctx.panel.state.base_view)
        logger.info(
            f"No base view found in panel state, using current view with {len(ctx.view)} samples"
        )
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

    def _handle_temporal_segments_field_change(self, ctx: Any) -> None:
        if "temporal_segments_field" in ctx.params:
            ctx.panel.state.temporal_segments_field = ctx.params[
                "temporal_segments_field"
            ]
        self._check_temporal_segments_populated(ctx)
        self._discover_segments(ctx)

    def _handle_temporal_segmentation_method_change(self, ctx: Any) -> None:
        """
        - Persist the selection method to ctx.panel.state
        """
        if "selection_method" in ctx.params:
            ctx.panel.state.temporal_segmentation_method = ctx.params["temporal_segmentation_method"]

    def _handle_propagation_method_change(self, ctx: Any) -> None:
        """
        - Persist the propagation method to ctx.panel.state
        """
        if "propagation_method" in ctx.params:
            ctx.panel.state.propagation_method = ctx.params[
                "propagation_method"
            ]

    def _check_temporal_segments_populated(self, ctx: Any) -> None:
        field = getattr(ctx.panel.state, "temporal_segments_field", None)
        if not field or field not in ctx.dataset.get_field_schema():
            ctx.panel.state.temporal_segments_populated = False
            return
        view = ctx.view
        has_segments = view.exists(field)
        ctx.panel.state.temporal_segments_populated = (
            len(has_segments) == len(view)
        )

    def _discover_segments(self, ctx: Any) -> None:
        """Build {segment_label: [sample_ids]} from temporal_segments classifications."""
        view = self.get_base_view(ctx)
        field = getattr(ctx.panel.state, "temporal_segments_field", None)
        if not field:
            return

        segment_to_ids = {}
        for sample in view:
            segs = sample.get_field(field)
            if segs and segs.classifications:
                for cls in segs.classifications:
                    lbl = cls.label
                    if lbl not in segment_to_ids:
                        segment_to_ids[lbl] = []
                    segment_to_ids[lbl].append(sample.id)

        ctx.panel.state.segments = segment_to_ids
        if segment_to_ids:
            ctx.ops.notify(
                f"Found {len(segment_to_ids)} segments in {field}",
                variant="success",
            )

    def _handle_input_annotation_field_change(self, ctx: Any) -> None:
        """
        - Persist the input annotation field to ctx.panel.state
        """
        if "input_annotation_field" in ctx.params:
            ctx.panel.state.input_annotation_field = ctx.params[
                "input_annotation_field"
            ]

    def _handle_output_annotation_field_change(self, ctx: Any) -> None:
        """
        - Persist the output annotation field to ctx.panel.state
        """
        if "output_annotation_field" in ctx.params:
            ctx.panel.state.output_annotation_field = ctx.params[
                "output_annotation_field"
            ]

    def _run_temporal_segmentation(self, ctx: Any) -> None:
        op_ctx = {
            "dataset": ctx.dataset,
            "view": ctx.view,
            "params": {
                "temporal_segments_field": getattr(
                    ctx.panel.state, "temporal_segments_field", "temporal_segments"
                ),
                "sort_field": getattr(ctx.panel.state, "sort_field", None),
                "temporal_segmentation_method": getattr(
                    ctx.panel.state, "temporal_segmentation_method", "heuristic"
                ),
            },
        }
        result = foo.execute_operator(
            "@51labs/label_propagation/temporal_segmentation", op_ctx
        )
        if result and hasattr(result, "result"):
            ctx.ops.notify(result.result.get("message", "Done"), variant="success")
        else:
            ctx.ops.notify("Temporal segmentation failed", variant="error")
        self._handle_temporal_segments_field_change(ctx)

    def _run_select_exemplars(self, ctx: Any) -> None:
        op_ctx = {
            "dataset": ctx.dataset,
            "view": ctx.view,
            "params": {
                "temporal_segments_field": getattr(
                    ctx.panel.state, "temporal_segments_field", "temporal_segments"
                ),
                "sort_field": getattr(ctx.panel.state, "sort_field", None),
                "exemplar_selection_method": "forward_only",  # TODO: Make this configurable
            },
        }
        result = foo.execute_operator(
            "@51labs/label_propagation/select_exemplars", op_ctx
        )
        if result and hasattr(result, "result"):
            ctx.ops.notify(result.result.get("message", "Done"), variant="success")
        else:
            ctx.ops.notify("Select exemplars failed", variant="error")
        self._handle_temporal_segments_field_change(ctx)

    def _handle_segment_selection(self, ctx: Any) -> None:
        if "selected_segment" in ctx.params:
            ctx.panel.state.selected_segment = ctx.params["selected_segment"]
        segment_label = getattr(ctx.panel.state, "selected_segment", None)
        if segment_label:
            try:
                propagation_view = self._create_propagation_view(ctx, segment_label)
                ctx.ops.set_view(propagation_view)
                ctx.ops.notify(
                    f"Opened propagation view for segment {segment_label}",
                    variant="info",
                )
            except Exception as e:
                logger.error(str(e), exc_info=True)
                ctx.ops.notify(f"Failed: {e}", variant="error")

    def _create_propagation_view(
        self, ctx: Any, segment_label: str
    ) -> fo.DatasetView:
        """Create propagation view for samples containing the given segment label."""
        segments = getattr(ctx.panel.state, "segments", {})
        base_view = self.get_base_view(ctx)
        if segment_label in segments:
            propagation_view = base_view.select(segments[segment_label])
        else:
            field = getattr(ctx.panel.state, "temporal_segments_field", None)
            if not field:
                raise RuntimeError("Temporal segments field not configured")
            propagation_view = base_view.match(
                {f"{field}.classifications": {"$elemMatch": {"label": segment_label}}}
            )

        if len(propagation_view) == 0:
            ctx.ops.notify(
                f"Empty propagation view for segment {segment_label}",
                variant="warning",
            )
        sort_field = getattr(ctx.panel.state, "sort_field", None)
        if sort_field and propagation_view.has_field(sort_field):
            propagation_view = propagation_view.sort_by(sort_field)
        return propagation_view

    def _run_propagate_labels(self, ctx: Any) -> None:
        """Execute PropagateLabels operator."""
        op_ctx = {
            "dataset": ctx.dataset,
            "view": ctx.view,
            "params": {
                "input_annotation_field": getattr(
                    ctx.panel.state, "input_annotation_field", None
                ),
                "output_annotation_field": getattr(
                    ctx.panel.state, "output_annotation_field", None
                ),
                "sort_field": getattr(ctx.panel.state, "sort_field", None),
                "propagation_method": getattr(
                    ctx.panel.state,
                    "propagation_method",
                    None,
                ),
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
            ctx.ops.notify(
                "Failed to run propagate_labels operator", variant="error"
            )

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
            "temporal_segments_field",
            label="Temporal Segments Field",
            default=getattr(ctx.panel.state, "temporal_segments_field", "temporal_segments"),
            description="Field storing temporal segment classifications",
            on_change=self._handle_temporal_segments_field_change,
        )

        populated = getattr(ctx.panel.state, "temporal_segments_populated", False)
        if populated:
            panel.md("#### Rerun Temporal Segmentation (Optional)", name="panel_rerun_header")
        else:
            panel.md("#### Temporal Segmentation", name="panel_segmentation_header")
        temporal_segmentation_method_dropdown = types.DropdownView()
        for choice in SUPPORTED_TEMPORAL_SEGMENTATION_METHODS:
            temporal_segmentation_method_dropdown.add_choice(choice, label=choice)
        panel.str(
            "temporal_segmentation_method",
            label="Segmentation Method",
            view=temporal_segmentation_method_dropdown,
            default=SUPPORTED_TEMPORAL_SEGMENTATION_METHODS[0],
            on_change=self._handle_temporal_segmentation_method_change,
        )
        panel.btn(
            "run_temporal_segmentation",
            label="Run Temporal Segmentation",
            on_click=self._run_temporal_segmentation,
            variant="contained",
        )
        panel.btn(
            "run_select_exemplars",
            label="Run Exemplar Selection",
            on_click=self._run_select_exemplars,
            variant="contained",
        )

        panel.md("#### Open Propagation View", name="panel_propagation_view_header")
        if populated:
            if not getattr(ctx.panel.state, "segments", {}):
                self._discover_segments(ctx)
            segments = getattr(ctx.panel.state, "segments", {})
            if segments:
                segment_dropdown = types.DropdownView()
                for seg_label, ids in segments.items():
                    segment_dropdown.add_choice(
                        seg_label, label=f"Segment {seg_label} [{len(ids)} samples]"
                    )
                panel.str(
                    "selected_segment",
                    label="Selected Segment",
                    view=segment_dropdown,
                    default=None,
                    on_change=self._handle_segment_selection,
                )
            else:
                panel.md("⚠️ No segments found. Run Temporal Segmentation first.", name="no_segments_warning")
        else:
            panel.md("⚠️ Temporal segments field not populated. Run Temporal Segmentation first.", name="field_not_populated_warning")

        # Propagation section
        panel.md("#### Propagation")
        panel.str(
            "input_annotation_field",
            label="Input Annotation Field",
            default=getattr(ctx.panel.state, "input_annotation_field", None),
            view=types.AutocompleteView(choices=field_choices)
            if field_choices
            else None,
            required=True,
            description="Field containing annotations to propagate from",
            on_change=self._handle_input_annotation_field_change,
        )
        input_annotation_field = getattr(
            ctx.panel.state, "input_annotation_field", None
        )
        if input_annotation_field:
            default_output_annotation_field = (
                input_annotation_field + "_propagated"
            )
        else:
            default_output_annotation_field = getattr(
                ctx.panel.state, "output_annotation_field", None
            )

        propagation_method_dropdown = types.DropdownView()
        for choice in SUPPORTED_PROPAGATION_METHODS:
            propagation_method_dropdown.add_choice(choice, label=choice)
        panel.str(
            "propagation_method",
            label="Propagation Method",
            view=propagation_method_dropdown,
            default=SUPPORTED_PROPAGATION_METHODS[0],
            description="Propagation method",
            on_change=self._handle_propagation_method_change,
        )

        panel.str(
            "output_annotation_field",
            label="Output Annotation Field",
            default=default_output_annotation_field,
            description=f"Field to store propagated annotations (default: {input_annotation_field}_propagated)",
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
