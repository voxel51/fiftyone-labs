# GetItem Refactor Notes

**Context:** The upstream FiftyOne repo (`fiftyone/utils/sam2.py`) has a `TODO` to
refactor `SegmentAnything2VideoModel` to use the `GetItem` pattern (see
`fiftyone/utils/torch.py`).  This file documents how our local caching work maps to
that future refactor so that whoever merges `sam2_local.py` with the upstream file
can carry the caching forward correctly.

---

## What GetItem will do

`GetItem` (defined in `fiftyone.utils.torch`) is an interface that separates
**data loading** from **model inference**:

```python
class GetItem:
    @property
    def required_keys(self) -> list[str]: ...   # what sample fields to read

    def __call__(self, d: dict) -> Any: ...      # load + preprocess one sample
```

`apply_model` calls `GetItem.__call__` for each sample, then passes the result
to the model's `predict` / `predict_all`.  For SAM2 image mode there are already
two concrete subclasses (`SegmentAnything2ImageGetItem`,
`SegmentAnything2ImageGetItemForVideo`).  The video model still loads frames
inline inside `predict` (inside `_forward_pass_boxes`), hence the upstream TODO.

---

## How our caching maps to GetItem

| Our current code | GetItem equivalent |
|---|---|
| `_compute_view_cache_key(parts)` | Key for a `GetItem`-result cache |
| `inference_state["images"]` | Tensor output of `GetItem.__call__` |
| `_backbone_cache[key][frame_idx]` | Pre-encoded features stored alongside the `GetItem` result |
| `_backbone_cache_context` | A caching wrapper around `GetItem.__call__` |

The two phases we introduced:

- **Phase 1 (inference_state cache)** — avoids re-running `load_video_frames` /
  creating the tmp symlink dir.  In GetItem terms, this is caching the *output* of
  `GetItem.__call__` keyed by view fingerprint.

- **Phase 2 (backbone cache)** — avoids re-running SAM2's `forward_image` (ViT
  backbone) for frames already seen.  In GetItem terms, this is caching
  pre-encoded features that would live *inside* `GetItem.__call__` (or alongside
  its return value).

---

## Steps to carry caching forward through the merge

### 1. Create `SAM2VideoGetItem`

When the upstream refactor happens, there will likely be a new class like:

```python
class SAM2VideoGetItem(GetItem):
    required_keys = ["id", "filepath", "frames"]

    def __call__(self, d: dict):
        # load + resize all frames → tensor
        images, height, width = load_video_frames(d["filepath"], ...)
        return {"images": images, "video_height": height, "video_width": width}
```

**Caching hook:** Wrap `__call__` with a keyed cache (similar to Phase 1):

```python
class CachingSAM2VideoGetItem(SAM2VideoGetItem):
    def __init__(self, ..., cache: dict | None = None):
        self._cache = cache if cache is not None else {}

    def __call__(self, d: dict):
        key = _compute_view_cache_key([f"{d['id']}:{d['filepath']}"])
        if key not in self._cache:
            self._cache[key] = super().__call__(d)
        return self._cache[key]
```

Pass the same `cache` dict across calls so state is shared.

### 2. Move backbone caching into `GetItem` or into a post-`GetItem` hook

The backbone features (`_backbone_cache`) are computed DURING `propagate_in_video`,
not during frame loading.  Two options for the GetItem world:

**Option A (eager):** Pre-encode all frames inside `GetItem.__call__` before
returning, and include the encoded features in the returned dict.  The model then
checks for pre-encoded features and skips `forward_image`.

```python
def __call__(self, d):
    images, h, w = load_video_frames(...)
    features = {fi: self.model.forward_image(images[fi]) for fi in range(len(images))}
    return {"images": images, "features": features, ...}
```

**Option B (lazy, current approach):** Keep `_backbone_cache_context` as a
context manager installed around `propagate_in_video`.  It intercepts
`_get_image_feature` and serves from cache.  This survives the GetItem refactor
unchanged — just move it from `SegmentAnything2VideoModel` into whatever class
wraps propagation.

Option B is lower-risk and requires no changes to the GetItem interface.

### 3. Cache invalidation

Currently the cache key is `md5("|".join(f"{id}:{filepath}" for each sample))`.
After the merge, keep this logic in `_compute_view_cache_key` — it's a static
method, easy to copy.  The cache is in-memory on the model object and lives for
the process lifetime, which is fine for the interactive operator use case.

If cross-process or cross-session persistence becomes desirable later, the cache
key is already designed to be stable (same view → same key across restarts).

### 4. Where the caches live after the merge

Currently they live on `SegmentAnything2VideoModel` (our local subclass).
After merging with the upstream class, put them on the same object (the model
instance).  The upstream `_SAM2_LOCAL_MODEL_CACHE` in `propagation.py` already
keeps the model alive between operator invocations, so the caches persist
correctly across calls.

---

## What to watch for during the merge

- `reset_state` clears per-object tracking but NOT `inference_state["images"]` or
  `inference_state["cached_features"]` (SAM2's 1-frame cache).  This is what
  makes Phase 1 safe.  Verify this is still true in the version you're merging
  against by reading `SAM2VideoPredictor.reset_state` and `_reset_tracking_results`.

- `_get_image_feature` (in `SAM2VideoPredictor`) replaces `cached_features` with a
  new single-entry dict on every call.  Our `_backbone_cache_context` works around
  this by pre-populating `cached_features` before each call.  If the upstream ever
  makes `cached_features` a proper LRU cache over all frames, Phase 2 becomes
  unnecessary (SAM2 would cache it itself).

- `_patch_sam2_memory_dtype_handling` (in `sam2_local.py`) also monkey-patches
  model methods.  Those patches are permanent (set once in `__init__`).  Our
  `_backbone_cache_context` patch is temporary (context manager).  Make sure both
  coexist: the dtype patches run on the *output* of `_run_single_frame_inference`
  and `_run_memory_encoder`, while the backbone patch replaces `_get_image_feature`
  which runs before `_run_single_frame_inference`.  No overlap.

---

## Local refactor completed (2026-06-24)

The local GetItem refactor has been applied to `sam2_local.py`.  Here is what
changed and what remains for the upstream merge.

### What was done

**New classes added** (before `SAM2ObjectTracker`):

- `SAM2VideoGetItem(fout.GetItem)` — base class.  `required_keys = ["sample",
  "video_reader"]`.  `__call__(d)` calls `predictor.init_state(image_folder)`
  or `predictor.init_state((sample, video_reader))` exactly as the old inline
  code did.  No caching.

- `CachingSAM2VideoGetItem(SAM2VideoGetItem)` — adds Phase 1
  inference_state caching.  Holds a **reference** to the owning model's
  `_inference_state_cache` dict (not a copy) so that `model._inference_state_cache`
  still reflects cache activity and callers can clear it directly.  `cache_key`
  is a plain `Optional[str]` attribute set externally before each call.

**Changes to `SegmentAnything2VideoModel`:**

- `__init__`: `_view_cache_key` instance attribute removed; replaced by
  `self._get_item = CachingSAM2VideoGetItem(self.model, self._inference_state_cache)`.

- `_view_cache_key` is now a property backed by `_get_item.cache_key`, so
  `propagation.py` (which sets `model._view_cache_key`) requires **no changes**.

- `_forward_pass_boxes` / `_forward_pass_points`: the old Phase 1
  if/else (`cache_key in self._inference_state_cache` … `init_state` …
  `self._inference_state_cache[key] = …`) is replaced by one line:
  ```python
  inference_state = self._get_item({"sample": sample, "video_reader": video_reader})
  ```
  Phase 2 `_backbone_cache_context` is unchanged (Option B, lazy).

### What still needs doing for the upstream merge

When merging with `fiftyone/utils/sam2.py`:

1. The upstream `SegmentAnything2VideoModel` already gets a `predict(video_reader,
   sample)` entry point.  You'll want to wire `_get_item` in the same place: create
   it in `__init__`, set `cache_key` from wherever the view fingerprint is managed,
   and call it at the top of `_forward_pass_boxes` / `_forward_pass_points`.

2. `SAM2VideoGetItem.required_keys = ["sample", "video_reader"]` is not
   standard field-name keys — these objects are passed directly in the dict rather
   than being looked up via `field_mapping`.  If the upstream wants true GetItem
   integration with `apply_model`'s field-extraction machinery, the keys would need
   to change (e.g., `"filepath"` + `"frames"`) and `__call__` would reconstruct the
   reader from those field values.  For now, the GetItem contract (separation of
   loading from inference) is satisfied without plugging into the field-mapping
   pipeline.

3. `_backbone_cache_context` stays on the model (Option B).  No changes needed
   there for the merge.
