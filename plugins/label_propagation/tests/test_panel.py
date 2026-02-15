import pytest
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.core.expressions import ViewField as F

# Import LabelPropagationPanel - add plugins directory to path
plugin_dir = Path(__file__).parent.parent
plugins_dir = plugin_dir.parent
if str(plugins_dir) not in sys.path:
    sys.path.insert(0, str(plugins_dir))
import label_propagation
LabelPropagationPanel = label_propagation.LabelPropagationPanel


@pytest.fixture
def dataset_view():
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    SELECT_SEQUENCES = ["bike-packing", "bmx-trees"]
    dataset_view = dataset.match_tags(SELECT_SEQUENCES)
    dataset_view = dataset_view.match(F("frame_number").to_int() < 3)

    if "labels_test" in dataset_view._dataset.get_field_schema():
        try:
            dataset_view._dataset.delete_sample_field(
                "labels_test", error_level=2
            )
        except AttributeError:
            assert (
                "labels_test" not in dataset_view._dataset.get_field_schema()
            ), "Unable to delete labels_test field"

        dataset_view._dataset.add_sample_field(
            "labels_test",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

    sequences = dataset_view.distinct("tags")
    sequences.remove("val")
    new_frame_number = 0
    for seq in sequences:
        seq_slice = dataset_view.match_tags(seq).sort_by("frame_number")
        seq_slice.set_values(
            "new_frame_number",
            [new_frame_number + ii for ii in range(len(seq_slice))],
        )
        new_frame_number += len(seq_slice)

        # label only the first
        exemplar_sample = seq_slice.first()
        exemplar_sample["labels_test"] = exemplar_sample["ground_truth"]
        exemplar_sample.save()

        # TODO(neeraja): support labeling an arbitrary frame

    return dataset_view


class MockPanelState:
    """Simple panel state object."""
    def __init__(self):
        self.exemplar_frame_field: Optional[str] = None
        self.sort_field: Optional[str] = None
        self.exemplar_field_exists_and_is_populated: bool = False
        self.exemplars: Dict[str, List[str]] = {}
        self.selected_exemplar: Optional[str] = None
        self.method: Optional[str] = None
        self.input_annotation_field: Optional[str] = None
        self.output_annotation_field: Optional[str] = None


class MockPanelOps:
    """Simple operations object to track view changes and notifications."""
    def __init__(self):
        self.view_changes = []
        self.notifications = []
    
    def set_view(self, view):
        self.view_changes.append(view)
    
    def notify(self, message, variant="info"):
        self.notifications.append({"message": message, "variant": variant})


class MockPanel:
    """Simple panel object."""
    def __init__(self):
        self.state = MockPanelState()
    
    def get_state(self, key):
        return getattr(self.state, key, None)


class MockPanelContext:
    """Panel context object with all required attributes."""
    def __init__(self, dataset, view, params=None):
        self.dataset = dataset
        self.view = view
        self.panel = MockPanel()
        self.params = params or {}
        self.ops = MockPanelOps()


def create_panel_context(dataset, view, params=None):
    """Create a panel context object with real FiftyOne objects."""
    return MockPanelContext(dataset, view, params)


class TestLabelPropagationPanel:
    def test_config_on_load(self, dataset_view):
        """Test on_load initializes panel state."""
        panel = LabelPropagationPanel()
        ctx = create_panel_context(dataset_view._dataset, dataset_view)
        panel.on_load(ctx)
        
        assert ctx.panel.state.exemplar_frame_field is None
        assert ctx.panel.state.sort_field is None
        assert ctx.panel.state.exemplar_field_exists_and_is_populated is False
        assert ctx.panel.state.exemplars == {}
        assert ctx.panel.state.selected_exemplar is None
    
    def test_handle_sort_field_change(self, dataset_view):
        """Test _handle_sort_field_change method."""
        panel = LabelPropagationPanel()
        ctx = create_panel_context(dataset_view._dataset, dataset_view, {"sort_field": "new_frame_number"})
        panel.on_load(ctx)
        
        panel._handle_sort_field_change(ctx)
        
        assert ctx.panel.state.sort_field == "new_frame_number"
        assert len(ctx.ops.view_changes) == 1
    
    def test_handle_exemplar_frame_field_change(self, dataset_view):
        """Test _handle_exemplar_frame_field_change method."""
        panel = LabelPropagationPanel()
        ctx = create_panel_context(dataset_view._dataset, dataset_view, {"exemplar_frame_field": "exemplar_test"})
        panel.on_load(ctx)
        
        panel._handle_exemplar_frame_field_change(ctx)
        
        assert ctx.panel.state.exemplar_frame_field == "exemplar_test"
    
    def test_check_exemplar_field_populated(self, dataset_view):
        """Test _check_exemplar_field_populated method."""
        panel = LabelPropagationPanel()
        ctx = create_panel_context(dataset_view._dataset, dataset_view)
        panel.on_load(ctx)
        
        # Test with non-existent field
        ctx.panel.state.exemplar_frame_field = "nonexistent"
        panel._check_exemplar_field_populated(ctx)
        assert ctx.panel.state.exemplar_field_exists_and_is_populated is False
        
        # Test with existing but unpopulated field
        dataset_view._dataset.add_sample_field(
            "exemplar_test_a",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.DynamicEmbeddedDocument,
        )
        ctx.panel.state.exemplar_frame_field = "exemplar_test_a"
        panel._check_exemplar_field_populated(ctx)
        assert ctx.panel.state.exemplar_field_exists_and_is_populated is False
    
    def test_run_assign_exemplar_frames(self, dataset_view):
        """Test _run_assign_exemplar_frames method."""
        panel = LabelPropagationPanel()
        ctx = create_panel_context(dataset_view._dataset, dataset_view)
        panel.on_load(ctx)
        ctx.panel.state.exemplar_frame_field = "exemplar_test"
        ctx.panel.state.sort_field = "new_frame_number"
        ctx.panel.state.method = "heuristic"
        
        panel._run_assign_exemplar_frames(ctx)
        # this calls _discover_exemplars and _handle_exemplar_frame_field_change

        assert isinstance(ctx.panel.state.exemplars, dict)
        assert ctx.panel.state.exemplar_field_exists_and_is_populated is True
        assert len(ctx.ops.view_changes) > 0
        assert len(ctx.ops.notifications) > 0
        assert any("success" in str(n.get("variant", "")) for n in ctx.ops.notifications)
    
    def test_handle_exemplar_selection(self, dataset_view):
        """Test _handle_exemplar_selection method."""
        panel = LabelPropagationPanel()
        ctx = create_panel_context(dataset_view._dataset, dataset_view)
        panel.on_load(ctx)
        
        # Set up exemplar field and discover exemplars
        ctx.panel.state.exemplar_frame_field = "exemplar_test"
        ctx.panel.state.sort_field = "new_frame_number"
        ctx.panel.state.method = "heuristic"
        panel._run_assign_exemplar_frames(ctx)
        
        # Select an exemplar
        if ctx.panel.state.exemplars:
            exemplar_id = list(ctx.panel.state.exemplars.keys())[0]
            ctx.params = {"selected_exemplar": exemplar_id}
            panel._handle_exemplar_selection(ctx)
            # this calls create_propagation_view

            assert ctx.panel.state.selected_exemplar == exemplar_id
            assert len(ctx.view) == len(ctx.panel.state.exemplars[exemplar_id])
            assert len(ctx.ops.view_changes) > 0

    def test_render(self, dataset_view):
        """Test render method returns Property."""
        panel = LabelPropagationPanel()
        ctx = create_panel_context(dataset_view._dataset, dataset_view)
        panel.on_load(ctx)
        
        result = panel.render(ctx)
        
        assert isinstance(result, types.Property)
    
    def test_run_propagate_labels(self, dataset_view):
        """Test _run_propagate_labels method."""
        panel = LabelPropagationPanel()
        ctx = create_panel_context(dataset_view._dataset, dataset_view)
        panel.on_load(ctx)
        ctx.panel.state.exemplar_frame_field = "exemplar_test"
        ctx.panel.state.sort_field = "new_frame_number"
        ctx.panel.state.method = "heuristic"
        panel._run_assign_exemplar_frames(ctx)
        ctx.panel.state.input_annotation_field = "labels_test"
        ctx.panel.state.output_annotation_field = "labels_test_propagated"
        panel._run_propagate_labels(ctx)
        assert len(ctx.ops.notifications) > 0
        assert any("success" in str(n.get("variant", "")) for n in ctx.ops.notifications)
