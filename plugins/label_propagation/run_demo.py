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


def get_video_from_images(dataset_view):
    import imageio
    dataset_view = dataset_view.sort_by("frame_number")
    images_dir = "/".join(dataset_view.first()["filepath"].split("/")[:-1])
    video_path = images_dir.replace("/JPEGImages/480p/", "/Videos/")
    video_path += ".mp4"
    if not os.path.exists(video_path):
        # write the images to a video
        with imageio.get_writer(video_path, fps=30) as writer:
            for image in dataset_view:
                writer.append_data(imageio.imread(image["filepath"]))
    return video_path


def make_video_dataset(image_dataset):
    image_dataset = image_dataset.group_by("sequence_id", order_by="frame_number")
    video_dataset = fo.Dataset()
    for sequence_id in image_dataset.values("sequence_id"):
        sequence_view = image_dataset.match_tags(sequence_id).sort_by("frame_number").flatten()
        video_path = get_video_from_images(sequence_view)
        video_dataset.add_sample(
            fo.Sample(
                filepath=video_path,
                tags=sequence_view.first()["tags"],
                sequence_id=sequence_id,
            )
        )
    video_dataset.ensure_frames()
    video_dataset.compute_metadata()

    frame_schema = image_dataset.get_field_schema()

    for sequence_id in video_dataset.values("sequence_id"):  # type: ignore
        video_sample = video_dataset.match_tags(sequence_id).first()
        sequence_view = image_dataset.match_tags(sequence_id).sort_by("frame_number").flatten()
        for image_sample, (frame_idx, frame) in zip(sequence_view, video_sample.frames.items()):
            for field_name in frame_schema.keys():
                if field_name in [
                    "id", "metadata", "created_at", "last_modified_at",
                    "filepath", "tags", "sequence_id", "frame_number",
                ]:
                    continue
                if image_sample[field_name] is None:
                    continue
                frame[field_name] = image_sample[field_name]
            frame.save()
    video_dataset.compute_metadata()
    return video_dataset


def get_dataset_view(num_scenes, media_format):
    dataset = foz.load_zoo_dataset(
        "https://github.com/voxel51/davis-2017",
        split="validation",
        format="image",
    )
    SELECT_SEQUENCES = ["car-roundabout", "car-shadow", "mbike-trick"]
    dataset_view = dataset.match_tags(SELECT_SEQUENCES[:num_scenes])
    dataset_view = dataset_view.match(F("frame_number").to_int() < 9)

    if media_format == "group":
        dataset_view = dataset_view.group_by("sequence_id", order_by="frame_number")
    if media_format == "video":
        dataset_view = make_video_dataset(dataset_view)
    
    return dataset_view


def get_partially_labeled_dataset_view(num_scenes, media_format):
    dataset_view = get_dataset_view(num_scenes, media_format)

    PURGE_FIELDS = ["labels_test", "labels_test_propagated", "frames.labels_test", "frames.labels_test_propagated"]
    for field_to_purge in PURGE_FIELDS:
        if field_to_purge in dataset_view._dataset.get_field_schema():
            dataset_view._dataset.delete_sample_field(
                field_to_purge, error_level=2
            )

    sequences = dataset_view.distinct("tags")
    sequences.remove("val")  # type: ignore
    new_frame_number = 0
    for seq in sequences:  # type: ignore

        if media_format == "video":
            video_sample = dataset_view.match_tags(seq).first()
            # label only the first
            exemplar_sample = video_sample.frames[1]
            exemplar_sample["labels_test"] = exemplar_sample["ground_truth"]
            video_sample.save()
        else:
            seq_view = dataset_view.match_tags(seq).sort_by("frame_number")
            seq_length = len(seq_view.flatten()) if media_format == "group" else len(seq_view)
            seq_view.set_values(
                "new_frame_number",
                [new_frame_number + ii for ii in range(seq_length)],
            )
            new_frame_number += seq_length

            # label only the first
            exemplar_sample = seq_view.first()
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
    parser.add_argument(
        "--media-format",
        type=str,
        default="image",
        help="Media format to use",
        choices=["image", "group", "video"],
    )
    args = parser.parse_args()

    view = get_partially_labeled_dataset_view(args.num_scenes, args.media_format)
    session = fo.launch_app(view)

    print("\n--------------------------------")
    print("Annotate some frames and propagate with the operator")
    print("--------------------------------\n")

    input("Press Enter to close the app...")
    session.close()
