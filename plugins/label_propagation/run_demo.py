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
    split="validation",
    format="image",
)
dataset.persistent = True

SELECT_SEQUENCES = ["dogs-jump"]
# TODO(neeraja): support multiple sequences
dataset = dataset.match_tags(SELECT_SEQUENCES)
dataset = dataset.match(F("frame_number").to_int() < 9)

session = fo.launch_app(dataset, remote=True)
webbrowser.open(f"http://localhost:5151")

print("\n--------------------------------")
print("Annotate some frames and propagate with the operator")
print("--------------------------------\n")

session.wait()
