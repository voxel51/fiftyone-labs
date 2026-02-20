# Label Propagation Plugin

Propagate annotations from sparsely labeled "exemplar frames" to all frames in a sequence using SAM-2.

This plugin exposes the following operator for use in the FiftyOne App and the Python SDK.
- `propagate_labels`
- `assign_exemplar_frames` (WIP)


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
   - **Name:** `propagate labels`
   - **Label:** *Propagate Labels From Input Field Operator*
5. Configure the presentaed field name options
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

## Operator: `assign_exemplar_frames`

Assigns exemplar frames to a view using selection methods (currently supports `heuristic`). Exemplar frames are key frames that represent distinct scenes or segments in a sequence.

### Parameters

- **`method`** (string, required)
  - Selection method: `"heuristic"` (detects scene discontinuities using image correlation)

- **`exemplar_frame_field`** (string, required, default: `"exemplar"`)
  - Field name for storing exemplar frame information
  - Creates subfields: `{field}.is_exemplar` (boolean) and `{field}.exemplar_assignment` (list of exemplar IDs)

- **`sort_field`** (string, optional)
  - Field used to sort samples before exemplar extraction

### Usage in the FiftyOne App

1. Open your dataset in the FiftyOne App.
2. Open the **Operators** dropdown and search for:
   - **Name:** `assign_exemplar_frames`
   - **Label:** *Assign Exemplar Frames Operator*
3. Configure the presentaed field name options
4. Run the operator
5. The `exemplar_frame_field` will appear on the bottom left with the subfield labels.

On success, you should see a message similar to:<br>
`Exemplar frames extracted and stored in field <exemplar_frame_field>`

---

## Interactive Panel: A Typical Workflow

The **Label Propagation** panel provides an interactive UI for the complete workflow:

1. Open the panel from the FiftyOne App sidebar
2. Configure the sort field (for image datasets)
3. **[Optional]** If an exemplar frame field exists and you want to use it, configure it to leverage the ability to interactively propagate through scenes.
3. **[Optional]** If an exemplar frame field does not exits, run `assign_exemplar_frames` to automatically select exemplar frames.
4. **[Optional]** Select an exemplar to open its propagation view (all frames assigned to that exemplar)
5. Label one or more frames in the propagation view, storing annotations in your chosen input field
6. Configure input and output annotation fields, then run `propagate_labels`
7. Inspect results and iterate as needed

The panel manages view state, exemplar discovery, and operator execution, streamlining the end-to-end workflow.

---

## ToDos

For the next PR

* [ ] Support backward propagation
* [ ] Add evaluation to the pytests
* [ ] Additional Exemplar selection methods

Product requirements

* [ ] Supports image datasets
* [ ] Supports video datasets
* [ ] Supports dynamically grouped datasets
* [ ] Propagated labels include instance IDs
* [ ] < 100ms per frame; faster for single-sample

Other features on the roadmap

* [ ] Single-sample (or few-sample) execution for interactive instance-wise propagation
* [ ] UX features that support HA (e.g. edit label field, "select" instances)
