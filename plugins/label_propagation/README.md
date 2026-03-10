# Label Propagation Plugin

Propagate annotations from sparsely labeled "exemplar frames" to all frames in a sequence using SAM-2.

This plugin exposes the following operators for use in the FiftyOne App and the Python SDK.

- `propagate_labels`
- `temporal_segmentation` — populates temporal segment classifications
- `select_exemplars` — sets exemplar scores on segment classifications

### Requirements

- **FiftyOne** installed and configured
- **SAM 2** installed:<br>
  `pip install "git+https://github.com/facebookresearch/segment-anything-2.git"`
- Network access for the first run:<br>
  The plugin will attempt to download SAM 2 weights (by default `sam2.1_hiera_tiny.pt`) into the installed SAM 2 package under `weights/` if they are not found.

---

## Operator: `propagate_labels`

- Takes a **view** over your dataset as input
- Looks for annotations in a given input field
- Uses SAM-2 to propagate those annotations to **all frames** in the view
- Writes the propagated annotations to a given output field

### Parameters

- **`input_annotation_field`** (string, required)

  - Sample-level field (for an image dataset) or Frame-level field (for a video dataset) containing the labels to propagate
  - Only frames where this field is **non-empty** are treated as exemplars

- **`output_annotation_field`** (string, required)

  - Sample-level field (for an image dataset) or Frame-level field (for a video dataset) where the propagated labels will be stored
  - **Must be different** from `input_annotation_field` to prevent accidental overwriting of ground truth annotations

- **`sort_field`** (string, optional)
  - **[For image datasets only]** Field used to sort samples before propagation, intended as a temporal index
  - If the view has this field, frames are ordered by it; otherwise, the operator falls back to the default dataset order

### Usage in the FiftyOne App

1. Open your dataset in the FiftyOne App.
2. Create a view containing the frames you want to process (for example, a subset of sequences or frames).
3. Ensure that:
   - Exemplar frames (_currently, must include the first frame of the sequence_) have labels in your chosen `input_annotation_field`.
4. Open the **Operators** dropdown and search for:
   - **Name:** `propagate_labels`
   - **Label:** `Propagate Labels From Input Field Operator`
5. Configure the presented field name options
6. Run the operator

On success, you should see a message similar to:<br>
`Annotations propagated from <input_field> to <output_field>`

### Notes and limitations

- The operator is designed for **image sequences / video frames** where temporal consistency is meaningful.
- Currently, fails on views with discontinuous scenes/labels.
- Currently, fails if annotated frames do not contain all relevant labels.
- Currently, only propagates forward in the sequence.
- On first run, downloading SAM 2 weights can take time, and GPU acceleration is strongly recommended for practical runtimes.

---

## Operator: `temporal_segmentation`

Populates a `fo.Classifications` field with temporal segment labels. Each classification has `label` (segment id) and `exemplar_score` (float, initially 0).

### Parameters

- **`temporal_segments_field`** (string, default: `"temporal_segments"`) — field to store classifications
- **`selection_method`** (string, default: `"heuristic"`) — detects scene discontinuities using image correlation
- **`sort_field`** (string, optional) — field to sort samples before segmentation

## Operator: `select_exemplars`

Sets `exemplar_score` on temporal segment classifications. For `forward_only`, the first frame of each segment gets score 1, others get 0.

### Parameters

- **`temporal_segments_field`** — same as above
- **`exemplar_selection_method`** (string, default: `"forward_only"`)
- **`sort_field`** (string, optional)

### Field schema

`{temporal_segments_field}` is `fo.Classifications`. Each `Classification` has:
- `label` (str) — segment identifier
- `exemplar_score` (float) — effectiveness as exemplar

---

## Interactive Panel: A Typical Workflow

1. Open the panel from the FiftyOne App sidebar
2. Configure sort field and temporal segments field
3. Run **Temporal Segmentation**, then **Exemplar Selection**
4. Select a segment label to open its propagation view (samples containing that label)
5. Label frames and run `propagate_labels`
6. Inspect results and iterate

---

## ToDos

For the next PR

- [ ] Support backward propagation
- [ ] Add evaluation to the pytests
- [ ] Additional Exemplar selection methods

Product requirements

- [ ] Supports image datasets
- [ ] Supports video datasets
- [ ] Supports dynamically grouped datasets
- [ ] Propagated labels include instance IDs
- [ ] < 100ms per frame; faster for single-sample

Other features on the roadmap

- [ ] Single-sample (or few-sample) execution for interactive instance-wise propagation
- [ ] UX features that support HA (e.g. edit label field, "select" instances)
