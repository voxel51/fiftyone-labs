import IPython
import os

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .ensemble_boxes_wbf import weighted_boxes_fusion


def box_combine_sample(sample, sources):
    box_list_all = []
    score_list_all = []
    label_list_all = []
    class_map = {}

    for s in sources:
        box_list = []
        score_list = []
        label_list = []
        for d in sample[s]:
            # Class index (dynamic).
            if not d["label"] in class_map.keys():
                class_map[d["label"]] = len(class_map.keys())
            label_list.append(class_map[d["label"]])
            # Bounding box.
            b = d["bounding_box"]
            b[2:] = [b[0] + b[2], b[1] + b[3]]  # convert x1y1whn to x1y1x2y2n
            box_list.append(b)
            # Model confidence.
            if d["confidence"] is None:
                score_list.append(1)
            else:
                score_list.append(d["confidence"])
        box_list_all.append(box_list)
        score_list_all.append(score_list)
        label_list_all.append(label_list)
    class_map_inverse = {class_map[i]: i for i in class_map.keys()}

    boxes, scores, labels = weighted_boxes_fusion(
        box_list_all, score_list_all, label_list_all
    )
    labels = [class_map_inverse[i] for i in labels]

    return boxes, scores, labels


def make_fo_detection(boxes, scores, labels):
    print("make detections")
    fo_detect = fo.Detections(detections=[])
    for b, s, l in zip(boxes, scores, labels):
        b[2:] = [b[2] - b[0], b[3] - b[1]]  # convert x1y1x2y2n to x1y1whn
        fo_detect["detections"].append(
            fo.Detection(label=l, bounding_box=b, confidence=s)
        )
    return fo_detect


class BoxCombine(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="box_combine", label="Box Combine Operator"
        )

    def resolve_input(self, ctx):
        print("resolve inputs")
        inputs = types.Object()
        if len(ctx.selected) > 0:
            inputs.str(
                "msg",
                view=types.Notice(
                    label=f"Combine Bounding Boxes on {len(ctx.selected)} Selected Samples"
                ),
            )
        else:
            inputs.str(
                "msg",
                view=types.Notice(
                    label=f"Combine Bounding Boxes on {len(ctx.view)} Samples"
                ),
            )
        active_fields = ctx.active_fields
        fields = ctx.view.get_field_schema(flat=True)
        detection_sources = [
            f
            for f in list(fields.keys())
            if ("detections" == f[-10:]) & (f.split(".")[0] in active_fields)
        ]

        field_str = ""
        output_name = "combine"
        for f in detection_sources:
            field_str += f"{f} "
            output_name += f[0]
        inputs.message(
            "message",
            "Label Sources Selected for Combination",
            description=field_str,
        )

        ctx.detection_sources = detection_sources
        ctx.output_name = output_name

        # TODO: add dropdown menu for NMS vs WBF.
        # TODO: add IoU threshold.

        return types.Property(inputs)

    def execute(self, ctx):
        print("execute")
        if len(ctx.selected) > 0:
            view = ctx.dataset.select(ctx.selected)
        else:
            view = ctx.view

        for s in view:
            boxes, scores, labels = box_combine_sample(
                s, ctx.detection_sources
            )
            fo_detect = make_fo_detection(boxes, scores, labels)
            s[ctx.output_name] = fo_detect
            s.save()

        ctx.trigger("reload_dataset")
        return {"new field": ctx.output_name, "updated": len(view)}

    def resolve_output(self, ctx):
        print("resolve output")
        outputs = types.Object()
        outputs.int("updated", label="Updated")
        outputs.str("new field", label="New Field")
        return types.Property(outputs)


def register(p):
    p.register(BoxCombine)
