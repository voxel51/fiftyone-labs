import fiftyone as fo

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