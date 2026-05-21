import contextlib

import fiftyone as fo
import fiftyone.core.storage as fos

try:
    import fiftyone.internal.context_vars as _ficv
    from fiftyone.internal.api_requests import (
        get_api_key_or_token as _get_api_key_or_token,
    )

    _TEAMS = True
except ImportError:
    _TEAMS = False


def get_auth() -> dict:
    """Return request_token/api_key kwargs for execute_operator (Teams only)."""
    if not _TEAMS:
        return {}
    api_key, request_token = _get_api_key_or_token()
    return {"request_token": request_token, "api_key": api_key}


@contextlib.contextmanager
def auth_context():
    """Ensure running_user_* context vars are populated before Teams RPC calls."""
    auth_dict = get_auth()
    api_key, request_token = auth_dict.get("api_key", None), auth_dict.get(
        "request_token", None
    )
    if api_key is None:
        yield
        return
    t = _ficv.running_user_request_token.set(request_token)
    k = _ficv.running_user_api_key.set(api_key)
    try:
        yield
    finally:
        _ficv.running_user_request_token.reset(t)
        _ficv.running_user_api_key.reset(k)


def get_frame_schema(ds: fo.Dataset) -> dict:
    if ds.media_type == "video":
        frame_level_schema = ds.get_frame_field_schema()
        frame_level_schema = {
            "frames." + k: v for k, v in frame_level_schema.items()  # type: ignore
        }
        return frame_level_schema
    else:
        return ds.get_field_schema()


def get_detections_fields(ds: fo.Dataset) -> dict:
    """Return only fields of type fo.Detections from the frame schema."""
    schema = get_frame_schema(ds)
    return {
        name: field
        for name, field in schema.items()
        if hasattr(field, "document_type")
        and field.document_type is fo.Detections
    }


def get_keypoints_fields(ds: fo.Dataset) -> dict:
    """Return only fields of type fo.Keypoints from the frame schema."""
    schema = get_frame_schema(ds)
    return {
        name: field
        for name, field in schema.items()
        if hasattr(field, "document_type")
        and field.document_type is fo.Keypoints
    }


def get_local_path(sample_or_frame):
    return (
        sample_or_frame.filepath
        if fos.is_local(sample_or_frame.filepath)
        else sample_or_frame.local_path
    )
