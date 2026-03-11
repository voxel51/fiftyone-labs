# Label Propagation Plugin

Propagate annotations from sparsely labeled "exemplar frames" to all frames in a sequence using SAM-2.

This plugin exposes the following operators for use in the FiftyOne App and the Python SDK.

- `propagate_labels` — uses SAM2 to use labels from an input field as prompts, to populate and output field
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

Populates an `fo.Classifications` field with temporal segment labels. Each classification has a `label` (segment id) and an `exemplar_score` (float, initially 0).

### Parameters

- **`temporal_segmentation_method`** (string, default: `"heuristic"`)
  - Presently, only supports a single method, based on sharp changes in RGB stats

- **`temporal_segments_field`**
  — Field in which to store classifications

- **`sort_field`** (string, optional)
  - **[For image datasets only]** Field used to sort samples before propagation, intended as a temporal index
  - If the view has this field, frames are ordered by it; otherwise, the operator falls back to the default dataset order

### Field schema

`{temporal_segments_field}` is `fo.Classifications`. Each `Classification` has:
- `label` (str) — segment identifier
- `exemplar_score` (float) — effectiveness as an exemplar of the segment

---

## Operator: `select_exemplars`

Only valid with an existing `fo.Classifications` field with temporal segment labels. Assigns an `exemplar_score` to each label of sample, indicating how valuable it is for this sample to be an exemplar for said temporal segment.

### Parameters

- **`temporal_segments_field`** (string, default: None)
  - Field whose classification labels to modify

- **`exemplar_scoring_method`** (string, default: `"first_frame"`) — detects scene discontinuities using image correlation
  - Depends on the Label Propagation method used.
  - Presently, only supports a single method, where the first frame of each segment gets score 1, others get 0. This is due to the lack of backward/bidirectional label propagation support.

- **`sort_field`** (string, optional)
  - **[For image datasets only]** Field used to sort samples before propagation, intended as a temporal index
  - If the view has this field, frames are ordered by it; otherwise, the operator falls back to the default dataset order

---

## Interactive Panel: A Typical Workflow

The **Label Propagation** panel provides an interactive UI for the complete workflow:

1. Open the panel from the FiftyOne App sidebar
2. **[For image datasets]** Configure the sort field for indicating the temporal sequence of images.
2. Configure the temporal segments field -- this may already exist, or will be where termporal detections are stored.
3. If the entire dataset does not belong to a single video scene (i.e., has discontinuities), run **Temporal Segmentation**. You can now select a segment label to open its propagation view (samples belonging to that temporal detection label)
4. Optionally, run **Exemplar Selection**. This will populate the `exemplar_score` field within the temporal segments classifications, to suggest how valuable of an exemplar each sample would be.
5. Label frames as needed.
6. Configure input and output annotation fields, then run `propagate_labels`
6. Inspect results and iterate

---

## ToDos

For the next PR

- [ ] Support backward propagation
- [ ] Add evaluation to the pytests
- [ ] Additional Temporal Segmentation methods

Product requirements

- [x] Supports image datasets
- [ ] Supports video datasets
- [ ] Supports dynamically grouped datasets
- [ ] Propagated labels include instance IDs
- [ ] < 100ms per frame; faster for single-sample

Other features on the roadmap

- [ ] Single-sample (or few-sample) execution for interactive instance-wise propagation
- [ ] UX features that support HA (e.g. edit label field, "select" instances)
- [ ] [Golden workflow](https://docs.google.com/document/d/1qbj5oqmaeEMF-LUE6jTO6hQtXAK2JqA2Sx3UC5F0RiI/edit?tab=t.xe2xwhs8joev)