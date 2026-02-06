# Image segmentation via point prompts

The `click_segmentation` panel is an interactive tool for generating point prompts via clicks. These point prompts can be input to a segmentation model that accepts point prompts (such as Segment Anything Model) to create segmentation masks.

## Usage

### Via FiftyOne App

_Select a sample and open Click To Segment panel_

<img src="assets/open_panel.png" alt="Click-To-Segment panel" width="600">

_In the Click To Segment panel, add a set of clicks for airplane and save as keypoints_

<img src="assets/clicks_airplane.png" alt="Keypoints for airplane" width="600">

_Add another set of clicks for sky and save as keypoints_

<img src="assets/clicks_sky.png" alt="Keypoints for sky" width="600">

_Choose a segmentation model from FiftyOne model zoo and click on Segment with Keypoints button_

<img src="assets/seg_success_notification.png" alt="Segmentation successful" width="600">

_Segmentation masks will be added to the Sample_

<img src="assets/output_segmentations.png" alt="Output segmentations" width="600">
