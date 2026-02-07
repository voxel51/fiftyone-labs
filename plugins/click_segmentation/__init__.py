import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone.zoo as foz


class SaveKeypoints(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_keypoints",
            label="Save Keypoints",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.list(
            "keypoints",
            types.List(types.Number()),
            required=True,
            label="Keypoints",
        )
        inputs.str("field_name", default="user_clicks", label="Field Name")
        inputs.str("label_name", default="label", label="Label Name")
        return types.Property(inputs)

    def execute(self, ctx):
        keypoints = ctx.params["keypoints"]
        field_name = ctx.params["field_name"]
        label_name = ctx.params["label_name"]

        dataset = ctx.dataset
        sample = dataset[ctx.current_sample]
        keypoint = fo.Keypoint(points=keypoints, label=label_name)

        if sample.has_field(field_name) and sample[field_name] is not None:
            num_kpts = len(sample[field_name].keypoints)
            ctx.ops.notify(
                f"Appending keypoints to {field_name} with {num_kpts} existing keypoint(s).",
                variant="warning",
            )
            sample[field_name].keypoints.append(keypoint)
        else:
            sample[field_name] = fo.Keypoints(keypoints=[keypoint])
        sample.save()
        sample.reload()


class SegmentWithPrompts(foo.Operator):
    _model_cache = None

    @property
    def config(self):
        return foo.OperatorConfig(
            name="segment_with_prompts",
            label="Segment With Prompts",
            description="Apply a promptable segmentation model to images",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
            allow_delegated_execution=True,
            allow_immediate_execution=True,
            default_choice_to_delegated=False,
            allow_distributed_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.view_target(ctx)

        inputs.str(
            "prompt_field",
            required=True,
            label="Prompt Field",
        )
        inputs.str(
            "label_field",
            required=False,
            label="Segmentation Label Field",
        )

        inputs.str(
            "model_name",
            required=True,
            label="Promptable Segmentation Model Name",
            default="segment-anything-2-hiera-small-image-torch",
        )

        inputs.int(
            "batch_size",
            label="Batch Size",
            description=("A batch size to use for segmentation"),
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
            inputs, view=types.View(label="Segment with prompts operator")
        )

    def _get_or_load_model(self, model_name):
        if self._model_cache is None or self._model_cache[0] != model_name:
            self._model_cache = [model_name, foz.load_zoo_model(model_name)]

        return self._model_cache[1]

    def execute(self, ctx):
        if ctx.current_sample:
            target_view = ctx.dataset.select(ctx.current_sample)
        else:
            target_view = ctx.target_view()

        prompt_field = ctx.params["prompt_field"]
        model_name = ctx.params["model_name"]
        label_field = ctx.params.get("label_field", prompt_field + "_seg")
        batch_size = ctx.params.get("batch_size", None)
        num_workers = ctx.params.get("num_workers", None)
        skip_failures = ctx.params.get("skip_failures", True)

        model = self._get_or_load_model(model_name)
        target_view.apply_model(
            model,
            label_field=label_field,
            prompt_field=prompt_field,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
        )
        target_view.save()

        ctx.ops.notify(
            f"Segmentation for prompts in {prompt_field} saved to {label_field}",
            variant="success",
        )

        if not ctx.delegated:
            ctx.ops.reload_dataset()


def register(p):
    p.register(SaveKeypoints)
    p.register(SegmentWithPrompts)
