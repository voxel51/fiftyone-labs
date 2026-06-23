"""Load the local label_propagation plugin for intensive tests."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import fiftyone as fo
from fiftyone.operators.decorators import cache as plugins_op_cache
from fiftyone.operators.decorators import dir_cache

_REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_PLUGINS_DIR = _REPO_ROOT / "plugins"


def _clear_plugins_cache() -> None:
    plugins_op_cache.clear()
    dir_cache["state"] = None


@pytest.fixture(autouse=True)
def use_local_plugins_dir():
    _clear_plugins_cache()
    with mock.patch.object(fo.config, "plugins_dir", str(LOCAL_PLUGINS_DIR)):
        yield
    _clear_plugins_cache()
