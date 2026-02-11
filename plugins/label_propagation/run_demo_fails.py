import os
import webbrowser

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

os.environ["VFF_EXP_ANNOTATION"] = "1"
print("\n--------------------------------")
print("Make sure your fiftyone installation is on the latest `develop`")
print("--------------------------------\n")


dataset = foz.load_zoo_dataset(
    "https://github.com/voxel51/davis-2017",
    split="train",
    format="image",
)
dataset.persistent = True

view1 = dataset.match_tags(["kid-football"])
view1 = view1.match(F("frame_number").to_int() < 24)
view1.set_values("new_frame_number", [ii for ii in range(len(view1))])

view2 = dataset.match_tags(["soccerball"])
view2 = view2.match(F("frame_number").to_int() < 8)
view2.set_values("new_frame_number", [len(view1) + ii for ii in range(len(view2))])
view = view1.concat(view2)


session = fo.launch_app(view, remote=True)
webbrowser.open(f"http://localhost:5151")

print("\n--------------------------------")
print("Annotate some frames and propagate with the operator")
print("--------------------------------\n")

session.wait()


"""
Traceback (most recent call last):
  File "/Users/neeraja/fiftyone/fiftyone/operators/executor.py", line 396, in execute_or_delegate_operator
    result = await do_execute_operator(
  File "/Users/neeraja/fiftyone/fiftyone/operators/executor.py", line 453, in do_execute_operator
    result = await (
  File "/Users/neeraja/fiftyone/fiftyone/core/utils.py", line 3137, in run_sync_task
    return await loop.run_in_executor(_get_sync_task_executor(), func, *args)
  File "/Users/neeraja/miniconda3/envs/fo/lib/python3.10/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/Users/neeraja/fiftyone/__plugins__/@51labs/label_propagation/__init__.py", line 150, in execute
    _ = propagate_annotations_sam2(
  File "/Users/neeraja/fiftyone/__plugins__/@51labs/label_propagation/sam2.py", line 465, in propagate_annotations_sam2
    _ = list(
  File "/Users/neeraja/fiftyone/fiftyone/core/collections.py", line 4614, in map_samples
    yield from mapper.map_samples(
  File "/Users/neeraja/fiftyone/fiftyone/core/map/mapper.py", line 167, in map_samples
    yield from self._map_samples(
  File "/Users/neeraja/fiftyone/fiftyone/core/map/mapper.py", line 275, in _map_samples
    raise err
  File "/Users/neeraja/fiftyone/fiftyone/core/map/mapper.py", line 238, in _map_samples_one_worker
    yield sample_id, None, map_fcn(sample)
  File "/Users/neeraja/fiftyone/__plugins__/@51labs/label_propagation/sam2.py", line 461, in populate_propagations
    sample[output_annotation_field] = propagated_detections
  File "/Users/neeraja/fiftyone/fiftyone/core/sample.py", line 71, in __setitem__
    super().__setitem__(field_name, value)
  File "/Users/neeraja/fiftyone/fiftyone/core/document.py", line 74, in __setitem__
    self.set_field(field_name, value)
  File "/Users/neeraja/fiftyone/fiftyone/core/sample.py", line 123, in set_field
    super().set_field(
  File "/Users/neeraja/fiftyone/fiftyone/core/document.py", line 806, in set_field
    super().set_field(
  File "/Users/neeraja/fiftyone/fiftyone/core/document.py", line 194, in set_field
    self._doc.set_field(
  File "/Users/neeraja/fiftyone/fiftyone/core/odm/mixins.py", line 176, in set_field
    self.add_implied_field(
  File "/Users/neeraja/fiftyone/fiftyone/core/odm/mixins.py", line 489, in add_implied_field
    field = create_implied_field(path, value, dynamic=dynamic)
  File "/Users/neeraja/fiftyone/fiftyone/core/odm/utils.py", line 303, in create_implied_field
    kwargs = get_implied_field_kwargs(value, dynamic=dynamic)
  File "/Users/neeraja/fiftyone/fiftyone/core/odm/utils.py", line 437, in get_implied_field_kwargs
    raise TypeError(
TypeError: Cannot infer an appropriate field type for value 'None'
"""