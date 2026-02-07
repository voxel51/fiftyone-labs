# Image segmentation via point prompts

Image segmentation via point prompts is a `fiftyone-labs` features that lets users interactively add keypoints to an image and use these keypoints for generating segmentation masks via a promptable segmentation model. There are two main components in this feaure:

1. The `click_segmentation` panel is an interactive tool for generating point prompts via clicks on the `Sample` modal. These point prompts can be input to a promptable segmentation model (such as Segment Anything Model) to create segmentation masks.

2. The `segment_wit_prompts` operator is a standalone operator that can be used to run a promptable segmentation model with either box or keypoint prompts saved in a sample field. The operator may be applied to the entire dataset or a subset of the dataset.

## Usage

### Click To Segment Panel

_Select a sample and open Click To Segment panel_

<img src="assets/open_panel.png" alt="Open panel" width="800">

_Enable capture mode by checking the box_

<img src="assets/click_capture_enabled.png" alt="Click-To-Segment panel" width="800">

The panel has two modes of operation:

- [Multiple Click Capture](#multi-click-mode): Users can save multiple sets of clicks and manually trigger segmentation.
- [Single Click Capture](#single-click-mode): Users can click on the sample modal and segmentation will auto-triggered.

#### Multiple Click Capture

_In the Sample modal, add a set of clicks for airplane and save as keypoints_

<img src="assets/clicks_airplane.png" alt="Keypoints for airplane" width="800">

_Add another set of clicks for sky and save as keypoints_

<img src="assets/clicks_sky.png" alt="Keypoints for sky" width="800">

NOTE: Clicked points are currently not rendered in the Sample modal. You can view them via the keypoints label field.

_Choose a segmentation model from FiftyOne model zoo and click on Segment with Keypoints button_

<img src="assets/seg_success_notification.png" alt="Segmentation successful" width="800">

_Segmentation masks will be added to the Sample_

<img src="assets/output_multi_click_segmentation.png" alt="Output segmentations for multiple clicks" width="800">

#### Single Click Capture

_In the Click To Segment panel, enable single click segmentation by checking the box_

<img src="assets/single_click_mode.png" alt="Single Click Segmentation" width="800">

_Add a click on the Sample modal to trigger auto-segmentation_

<img src="assets/output_single_click_segmentation.png" alt="Keypoints for sky" width="800">
