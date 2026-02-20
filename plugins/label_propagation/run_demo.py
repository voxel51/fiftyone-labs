import os
import argparse

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

os.environ["VFF_EXP_ANNOTATION"] = "1"
print("\n--------------------------------")
print("Make sure your fiftyone installation is on the latest `develop`")
print("--------------------------------\n")


def get_dataset_view(num_scenes):
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    SELECT_SEQUENCES = ["car-roundabout", "car-shadow", "mbike-trick"]
    dataset_view = dataset.match_tags(SELECT_SEQUENCES[:num_scenes])
    dataset_view = dataset_view.match(F("frame_number").to_int() < 9)
    return dataset_view


def get_partially_labeled_dataset_view(num_scenes):
    dataset_view = get_dataset_view(num_scenes)

    if "labels_test" in dataset_view._dataset.get_field_schema():
        try:
            dataset_view._dataset.delete_sample_field(
                "labels_test", error_level=2
            )
        except AttributeError:
            assert (
                "labels_test" not in dataset_view._dataset.get_field_schema()
            ), "Unable to delete labels_test field"

    if "labels_test" not in dataset_view._dataset.get_field_schema():
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

    return dataset_view


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=1,
        help="Number of sequences to include",
    )
    args = parser.parse_args()

    view = get_partially_labeled_dataset_view(args.num_scenes)
    session = fo.launch_app(view)

    print("\n--------------------------------")
    print("Annotate some frames and propagate with the operator")
    print("--------------------------------\n")

    input("Press Enter to close the app...")
    session.close()
