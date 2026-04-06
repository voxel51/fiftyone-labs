# Weighted Boxes Fusion

A plugin that provides methods for ensembling bounding boxes from different
object detection models directly within the
[FiftyOne App](https://docs.voxel51.com), using [Weighted Boxes Fusion (WBF)](https://arxiv.org/abs/1910.13302).

Based on [ZFTurbo's Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion).

## Installation

```shell
fiftyone labs install @51labs/box_combine
```

Refer to the
[main README](https://github.com/voxel51/fiftyone-plugins) for more
information about managing downloaded plugins and developing plugins locally.

## Usage

1. Launch the App:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

1. Click the `Browse operations` action to open the
   Operators list
2. Select the `box_combine` operator.

### Example workflow

A common workflow is to run multiple object detection models on a dataset and
then combine their predictions:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")

# Run multiple detection models
model1 = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
model2 = foz.load_zoo_model("retinanet-resnet50-fpn-coco-torch")

dataset.apply_model(model1, label_field="det_faster_rcnn")
dataset.apply_model(model2, label_field="det_retinanet")

session = fo.launch_app(dataset)
```

In the app, select `det_faster_rcnn` and `det_retinanet` labels in the panel on the left. Open `box_combine` operator to fuse the predictions from
the selected label into a single field using weighted boxes fusion method.
