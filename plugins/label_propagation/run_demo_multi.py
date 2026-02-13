import os

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
view1.set_values(
  "new_frame_number", [ii for ii in range(len(view1))]
)

view2 = dataset.match_tags(["soccerball"])
view2 = view2.match(F("frame_number").to_int() < 24)
view2.set_values(
    "new_frame_number", [len(view1) + ii for ii in range(len(view2))]
)
view = view1.concat(view2)


session = fo.launch_app(view)

print("\n--------------------------------")
print("Annotate some frames and propagate with the operator")
print("--------------------------------\n")

input("Press Enter to close the app...")
session.close()
