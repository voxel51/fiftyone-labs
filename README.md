# FiftyOne Labs

<p align="center">
  <img src="assets/labs_logo_full_dark.png" alt="FiftyOne Labs Logo" width="50%">
</p>

FiftyOne Labs brings research solutions and experimental features for machine learning.

## Table of Features

This repository contains a curated collection of
FiftyOne Labs Features which are developed using the [FiftyOne plugins ecosystem](https://docs.voxel51.com/plugins/index.html). These features are organized into the following categories:

- [Machine Learning Lab](#ml-features): core machine learning experimental features
- [Visualization Lab](#visualization-features): features for advanced visualization

## Machine Learning Lab

<table>
    <tr>
        <th>Name</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/labs/tree/main/plugins/labs_panel">@51labs/labs_panel</a></b></td>
        <td>A panel listing all available Labs plugins</td>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/labs/tree/main/plugins/video_apply_model">@51labs/video_apply_model</a></b></td>
        <td>Apply image model to video dataset using torch dataloader</td>
    </tr>
    <tr>
        <td><b><a href="https://github.com/griffbr/box-combine">@griffbr/box-combine</a></b></td>
        <td>Weighted Box Fusion for detections</td>
    </tr>
    <tr>
        <td><b><a href="https://github.com/voxel51/zero-shot-coreset-selection">@voxel51/zero-shot-coreset-selection</a></b></td>
        <td>Zero-shot coreset selection (ZCore) for unlabeled image data</td>
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

## Using Labs

### Install FiftyOne

If you haven't already, install
[FiftyOne](https://github.com/voxel51/fiftyone):

```shell
pip install fiftyone
```

### Installing a Labs Feature

To install all the features in this repository, you can run:

```shell
fiftyone plugins download https://github.com/voxel51/labs
```

You can also install a specific plugin using the `--plugin-names` flag:

```shell
fiftyone plugins download \
    https://github.com/voxel51/labs \
    --plugin-names <name1> <name2> <name3>
```

### Installing via Labs Panel

[Labs Panel](plugins/labs_panel/README.md) offers a convenient interface to install Labs features in the FiftyOne App. To get started, install the Labs Panel:

```shell
fiftyone plugins download \
    https://github.com/voxel51/fiftyone-plugins \
    --plugin-names @labs/labs_panel
```

## Contributing

Check out the [contributions guide](CONTRIBUTING.md) for more information.
