import fiftyone as fo
import fiftyone.core.storage as fos


def get_frame_schema(ds: fo.Dataset) -> dict:
    if ds.media_type == "video":
        frame_level_schema = ds.get_frame_field_schema()
        frame_level_schema = {
            "frames." + k: v
            for k, v in frame_level_schema.items()  # type: ignore
        }
        return frame_level_schema
    else:
        return ds.get_field_schema()


def get_local_path(sample_or_frame):
    return (
        sample_or_frame.filepath
        if fos.is_local(sample_or_frame.filepath)
        else sample_or_frame.local_path
    )
