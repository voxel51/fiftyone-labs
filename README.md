# FiftyOne Labs

<div align="center">
<p align="center">
  <img src="https://github.com/voxel51/labs/raw/main/assets/labs_logo_transparent_light.svg#gh-light-mode-only" alt="FiftyOne Labs Logo" width="50%">
  <img src="https://github.com/voxel51/labs/raw/main/assets/labs_logo_transparent_dark.svg#gh-dark-mode-only" alt="FiftyOne Labs Logo" width="50%">

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</p>
</div>

FiftyOne Labs brings research solutions and experimental features for machine learning.

## Table of Features

This repository contains a curated collection of
FiftyOne Labs Features which are developed using the [FiftyOne plugins ecosystem](https://docs.voxel51.com/plugins/index.html). These features are organized into the following categories:

- [Machine Learning Lab](#machine-learning-lab): core machine learning experimental features
- [Visualization Lab](#visualization-lab): features for advanced visualization

## Machine Learning Lab

<table>
    <tr>
        <th>Name</th>
        <th>Tags</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/fiftyone-labs/tree/main/plugins/labs_panel">@51labs/labs_panel</a></b></td>
        <td><kbd>ml</kbd> <kbd>utils</kbd></td>
        <td>A panel listing all the available FiftyOne Labs features</td>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/fiftyone-labs/tree/main/plugins/video_apply_model">@51labs/video_apply_model</a></b></td>
        <td><kbd>ml</kbd> <kbd>video</kbd></td>
        <td>Apply image model to video dataset using torch dataloader</td>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/fiftyone-labs/tree/main/plugins/few_shot_learning">@51labs/few_shot_learning</a></b></td>
        <td><kbd>ml</kbd> <kbd>classification</kbd></td>
        <td>Interactive few-shot learning with multiple model types</td>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/fiftyone-labs/tree/main/plugins/label_propagation">@51labs/label_propagation</a></b></td>
        <td><kbd>ml</kbd> <kbd>video</kbd> <kbd>segmentation</kbd></td>
        <td>Propagating Labels across frames of a video</td>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/fiftyone-labs/tree/main/plugins/box_combine">@51labs/box_combine</a></b></td>
        <td><kbd>ml</kbd> <kbd>detection</kbd></td>
        <td>Weighted Boxes Fusion for detections</td>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/zero-shot-coreset-selection">@51labs/zero-shot-coreset-selection</a></b></td>
        <td><kbd>ml</kbd></td>
        <td>Zero-shot coreset selection (ZCore) for unlabeled image data</td>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/fiftyone-labs/tree/main/plugins/click_segmentation">@51labs/click_segmentation</a></b></td>
        <td><kbd>ml</kbd> <kbd>segmentation</kbd></td>
        <td>Interactive image segmentation via prompts</td>
    </tr>
</table>

## Visualization Lab

<table>
    <tr>
        <th>Name</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>@51labs/viz_placeholder</td>
        <td>Placeholder for visualization feature</td>
    </tr>
</table>

## Using FiftyOne Labs

### Install FiftyOne

If you haven't already, install
[FiftyOne](https://github.com/voxel51/fiftyone):

```shell
pip install fiftyone
```

### Installing specific FiftyOne Labs Feature

To install all the features in this repository, you can run:

```shell
fiftyone labs install --all
```

You can also install specific FiftyOne Labs features using:

```shell
fiftyone labs install <name1> <name2> ...
```

### Installing via Labs Panel

[Labs Panel](plugins/labs_panel/README.md) offers a convenient interface to install FiftyOne Labs features in the FiftyOne App. To get started, install the Labs Panel:

```shell
fiftyone labs install @51labs/labs_panel
```

### FiftyOne Labs CLI

For more command line tools for FiftyOne Labs, check out the [CLI documentation](https://docs.voxel51.com/cli/index.html#fiftyone-labs).

## Feedback

For questions, comments, and suggestions, head to the `fiftyone-labs` [Discord Channel](https://discord.com/channels/1266527359511564372/1466492755214733625).

## Contributing

Check out the [contributions guide](CONTRIBUTING.md) for more information.
