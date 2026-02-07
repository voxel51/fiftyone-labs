# Few-Shot Learning Plugin

Interactive few-shot learning panel for FiftyOne that lets you train classifiers by labeling just a few positive and negative samples.

## Features

- **Multiple embedding models**: Choose from ResNet18, ResNet50, CLIP (ViT-B/32), or DINOv2 (ViT-B/14) with automatic embedding computation
- **Interactive labeling**: Select samples in the grid and label them with one click
- **Iterative refinement**: Train, review predictions, add more labels, repeat
- **Subset sampling**: Limit inference to a working subset for faster iteration on large datasets
- **Fast inference**: Uses PyTorch DataLoaders for efficient batch processing

## Supported Models

| Model                     | Description                         | Best For                     |
| ------------------------- | ----------------------------------- | ---------------------------- |
| **RocchioPrototypeModel** | Centroid-based prototype classifier | Balanced data, interpretable |

## Prerequisites

The plugin ships with self-contained model implementations. Required Python dependencies are:

- `numpy`
- `torch`

## Installation

```bash
fiftyone plugins download \
    https://github.com/voxel51/labs \
    --plugin-names @51labs/few_shot_learning
```

## Usage

### Quick Start

Run the demo script to launch with a sample dataset:

```bash
cd /path/to/fiftyone-labs
python plugins/few_shot_learning/run_demo.py
```

### Panel Workflow

The panel has two screens: a **setup screen** for configuring a new session, and an **active session screen** for labeling and training.

#### 1. Configure and Start a Session

When no session is active, the panel shows configuration options:

- **Embedding Model**: Select from ResNet18, ResNet50, CLIP (ViT-B/32), or DINOv2 (ViT-B/14)
- **Embedding Field**: Auto-populated based on the selected model (e.g. `resnet18_embeddings`), but can be customized
- **Advanced Settings**: Batch size, number of DataLoader workers, and skip-failures toggle
- **Subset Sampling**: Optionally limit inference to a fixed number of samples

Click **Start Session** to initialize. The plugin validates embedding dimensions, creates training label fields (`train_positive`, `train_negative`), and computes embeddings for any samples that don't have them yet.

#### 2. Label Samples

Once a session is active, the panel header shows the current iteration count and label totals.

1. Select samples in the FiftyOne grid
2. Click **Label Positive** or **Label Negative**

Labels are persistent across iterations. If you re-label a sample (e.g. switch from positive to negative), the old label is removed automatically.

#### 3. Train and Predict

You need at least 1 positive and 1 negative sample to train. Select a classifier model and configure its hyperparameters, then click **Train & Label Dataset**.

The training step:

1. Ensures embeddings exist for all labeled and inference samples
2. Collects labeled embeddings, L2-normalizes them, and fits the model
3. Runs inference on all samples (or the working subset) via a PyTorch DataLoader
4. Writes predictions to the `fewshot_prediction` field as `fo.Classification` labels with confidence scores
5. Sets the view to show predicted samples sorted positive-first

#### 4. Iterate

Review the predictions, select misclassified or uncertain samples, relabel them, and click **Train & Label Dataset** again. Each click increments the iteration counter.

#### 5. Reset

Click **Reset Session** to clear all session state and delete the `fewshot_prediction`, `train_positive`, and `train_negative` fields from the dataset.

### Workflow Tips

- Start with clear examples: label the most obvious positive and negative samples first
- After training, the view is sorted with positives first — scroll down to find false positives to relabel as negative
- Adjust model hyperparameters between iterations to tune behavior

## Configuration

### Embeddings

| Display Name      | Zoo Model                  | Field Default              | Dimensions |
| ----------------- | -------------------------- | -------------------------- | ---------- |
| ResNet18          | `resnet18-imagenet-torch`  | `resnet18_embeddings`      | 512        |
| ResNet50          | `resnet50-imagenet-torch`  | `resnet50_embeddings`      | 2048       |
| CLIP (ViT-B/32)  | `clip-vit-base32-torch`    | `clip_vit_b32_embeddings`  | 512        |
| DINOv2 (ViT-B/14)| `dinov2-vitb14-torch`      | `dinov2_vitb14_embeddings` | 768        |

The embedding field name auto-updates when you change the model, but only if you haven't set a custom name. Embeddings are computed automatically for any samples missing them.

If the selected field already contains embeddings with a different dimension than expected, the plugin shows an error and blocks session start.

### Model Hyperparameters

Hyperparameters are configured directly in the panel UI during an active session:

**RocchioPrototypeModel**:
- `mode`: `proto_softmax` (nearest centroid scoring) or `rocchio_sigmoid` (query vector dot product). Default: `proto_softmax`
- `beta`: Weight for positive centroid. Default: `1.0`
- `gamma`: Weight for negative centroid. Default: `1.0`
- `temperature`: Temperature for softmax/sigmoid scaling. Default: `1.0`

### Advanced Settings

- **Batch Size**: Number of samples per inference batch (default: 16)
- **Num Workers**: DataLoader workers for parallel loading (default: 8)
- **Skip Failures**: Skip samples that fail to load (default: true)

### Subset Sampling

For large datasets, you can limit inference to a working subset:

- **Working Subset Size**: Set to N > 0 to run inference on only N samples. Labeled samples are always included regardless of the limit. Set to 0 (default) to use all samples.
- **Randomize Each Iteration**: When enabled, a new random subset is sampled each time you click **Train & Label Dataset**. When disabled, the subset is cached and reused until the underlying view changes.
