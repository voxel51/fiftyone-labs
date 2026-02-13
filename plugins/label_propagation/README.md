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

- **`sort_field`** (string, optional, default: `"frame_number"`)
  - **[For image datasets only]** Field used to sort samples before propagation, intended as a temporal index
  - If the view has this field, frames are ordered by it; otherwise, the operator falls back to the default dataset order


### Usage in the FiftyOne App

1. Open your dataset in the FiftyOne App.
2. Create a view containing the frames you want to process (for example, a subset of sequences or frames).
3. Ensure that:
   - Exemplar frames (_currently, must include the first frame of the sequence_) have labels in your chosen `input_annotation_field`.
4. Open the **Operators** panel and search for:
   - **Name:** `propagate_labels`
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

## Typical workflow

1. Select the view of samples to be processed.
2. If this is an image dataset, sort images temporally. Ensure that the sort order is stored in some sample field. This will be the `sort_field` requested by the `propagate_labels` operator.
3. **[Optional]** Run `assign_exemplar_frames` to get `{exemplar_frame_field}.is_exemplar` labels on samples/frames.
4. Label one or more frames from the view, and store them in a sample-level field (for an image dataset) or frame-level field (for a video dataset). This will be your `input_annotation_field` for input in the `propagate_labels` operator.
5. Decide on a target field for propagated labels which does not clash with the input field.
6. Run `propagate_labels` in the App or via Python, on the desired view.
7. Inspect reesults. If necessary, edit/add more annotations.

---

## ToDos

For this PR

* [x] Add `assign_exemplar_frames`
* [x] Add a demo script using HA
* [ ] Add evaluation to the intensive pytest
* [ ] Documentation in Confluence


Product requirements

- [x] Supports image datasets
- [ ] Supports video datasets
- [ ] Supports dynamically grouped datasets
- [ ] Propagated labels include **instance IDs**
- [ ] \< 100ms per frame; faster for single-sample


Other features on the roadmap
- [ ] Add a panel for interactively viewing exemplars + scene-wise results
- [ ] Support propagation backward in time
- [ ] Single-sample (or few-sample) execution for interactive instance-wise propagation
- [ ] UX features that support HA (e.g. edit label field, "select" instances)