"""Operator to apply image model to video frames using a dataloader.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone.operators as foo
import fiftyone.zoo as foz
from fiftyone.operators import types

from .model_inference import apply_image_model_to_video_frames


class VideoApplyModel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="video_apply_model",
            label="Apply model to video frames",
            description="Apply an image model to frames of a video",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
            allow_delegated_execution=True,
            allow_immediate_execution=True,
            default_choice_to_delegated=True,
            allow_distributed_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.view_target(ctx)

        # Apply model parameters
        inputs.str("model", label="Image Model", required=True)
        inputs.str(
            "label_field",
            required=True,
            label="Label field",
            description=(
                "The name of field in which to store the predictions"
            ),
        )
        inputs.float(
            "conf_thresh", label="Confidence Threshold", required=False
        )
        inputs.int(
            "batch_size",
            label="Batch Size",
            description=("A batch size to use for videos"),
            required=False,
            default=1,
        )
        inputs.int(
            "num_workers",
            label="Num Workers",
            description=(
                "The number of workers to use for torch dataloaders "
                "(if applicable)"
            ),
            required=False,
        )

        inputs.int(
            "frames_chunk_size",
            label="Chunk size for video frames",
            description=(
                "The number of frames to read in one chunk for a video"
            ),
            required=False,
        )

        inputs.str(
            "parse_type",
            label="Parsing Type",
            description=(
                "Whether to parse chunks in an interleaved manner from videos or "
                "parse chunks sequentially from one video before moving to the next video"
            ),
            required=False,
            default="sequential",
        )

        inputs.bool(
            "skip_failures",
            label="Skip Failures",
            description=(
                "Whether to gracefully continue without raising an error if "
                "prediction cannot be generated for a sample"
            ),
            required=False,
        )

        return types.Property(
            inputs, view=types.View(label="Video Apply Model operator")
        )

    def execute(self, ctx):
        target_view = ctx.target_view()

        model_name = ctx.params.get("model")
        model = foz.load_zoo_model(model_name)

        label_field = ctx.params.get("label_field", None)
        conf_thresh = ctx.params.get("conf_thresh", None)
        batch_size = ctx.params.get("batch_size", None)
        num_workers = ctx.params.get("num_workers", None)
        frames_chunk_size = ctx.params.get("frames_chunk_size", None)
        parse_type = ctx.params.get("parse_type", None)
        skip_failures = ctx.params.get("skip_failures", True)

        apply_image_model_to_video_frames(
            target_view,
            model,
            label_field=label_field,
            confidence_thresh=conf_thresh,
            batch_size=batch_size,
            frames_chunk_size=frames_chunk_size,
            parse_type=parse_type,
            num_workers=num_workers,
            skip_failures=skip_failures,
            progress=None,
        )

        if not ctx.delegated:
            ctx.trigger("reload_dataset")


def register(p):
    p.register(VideoApplyModel)
