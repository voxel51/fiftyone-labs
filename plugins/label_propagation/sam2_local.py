"""
Local SAM2 wrapper used by the `label_propagation` plugin.

This module exists so we can extend FiftyOne's SAM2 behavior locally (without
modifying the FiftyOne repo) in order to support the prompt / frame loading
patterns used by our Davis propagation tests.
"""

import logging
import os
import tempfile
from typing import Any, Iterable, List, Optional, cast

import cv2
import eta.core.utils as etau
import numpy as np

import fiftyone.core.labels as fol
import fiftyone.core.media as focm
import fiftyone.core.models as fom
import fiftyone.core.utils as fou
import fiftyone.core.storage as fos
import fiftyone.utils.sam as fosam
from fiftyone.utils.sam2 import (
    SegmentAnything2VideoModel as FiftyOneSegmentAnything2VideoModel,
)
import fiftyone.utils.torch as fout
import fiftyone.zoo.models as fozm

from .utils import get_local_path

logger = logging.getLogger(__name__)

fou.ensure_torch()
import torch  # noqa: E402

sam2 = fou.lazy_import("sam2", callback=lambda: fou.ensure_import("sam2"))
samg = fou.lazy_import("sam2.automatic_mask_generator")
smip = fou.lazy_import("sam2.sam2_image_predictor")
smutil = fou.lazy_import("sam2.utils.misc")


def _to_abs_mask(mask, abs_box, img_width, img_height):
    """
    Args:
        mask: numpy array relative to box
        abs_box: [x1, y1, x2, y2]
        img_width: width of the image
        img_height: height of the image

    Returns:
        numpy array relative to the image
    """
    x1, y1, x2, y2 = abs_box
    box_width = x2 - x1
    box_height = y2 - y1
    mask_fitted = np.pad(
        mask,
        [
            (0, max(0, box_height - mask.shape[0])),
            (0, max(0, box_width - mask.shape[1])),
        ],
    )[:box_height, :box_width]
    mask_framed = np.zeros((img_height, img_width), bool)
    mask_framed[y1:y2, x1:x2] = mask_fitted
    return mask_framed.astype(np.uint8)


def _subtract_negative_box_regions(
    mask: np.ndarray,
    neg_detections: Optional[fol.Detections],
    width: int,
    height: int,
) -> np.ndarray:
    """Subtract negative bounding box regions from a segmentation mask.

    Args:
        mask: numpy array representing segmentation mask(s). Supports:
            - 2D (H, W) for video model
            - 3D (N, H, W) for image model
            - 4D (N, 1, H, W) for image model
        neg_detections: a :class:`fiftyone.core.labels.Detections` containing
            negative prompt boxes, or None
        width: image/frame width in pixels
        height: image/frame height in pixels

    Returns:
        the modified mask with negative regions zeroed out
    """
    if neg_detections is None:
        return mask

    detections = cast(Iterable[fol.Detection], neg_detections.detections)
    for neg_det in detections:
        box_xyxy = fosam._to_abs_boxes(
            np.array([neg_det.bounding_box]), width, height, chunk_size=1
        )
        box_abs = np.round(box_xyxy.squeeze()).astype(int)
        nx1, ny1, nx2, ny2 = (
            max(0, box_abs[0]),
            max(0, box_abs[1]),
            min(width, box_abs[2]),
            min(height, box_abs[3]),
        )
        if nx2 > nx1 and ny2 > ny1:
            if mask.ndim == 2:
                mask[ny1:ny2, nx1:nx2] = 0
            elif mask.ndim == 3:
                mask[:, ny1:ny2, nx1:nx2] = 0
            else:
                mask[:, 0, ny1:ny2, nx1:nx2] = 0

    return mask


class SegmentAnything2VideoModelConfig(
    fout.TorchImageModelConfig, fozm.HasZooModel
):
    """Configuration for running a :class:`SegmentAnything2VideoModel`.

    See :class:`fiftyone.utils.torch.TorchImageModelConfig` for additional
    arguments.

    Args:
        media_mode (None): the media mode to use for the model (only "video"
            and "image" are supported). If None, defaults to "video"
    """

    def __init__(self, d):
        d = self.init(d)
        super().__init__(d)

        self.media_mode = self.parse_string(d, "media_mode", default="video")  # type: ignore[arg-type]
        if self.media_mode not in ["video", "image"]:
            raise ValueError("media_mode must be one of 'video' or 'image'")


class SegmentAnything2VideoModel(FiftyOneSegmentAnything2VideoModel):
    """Local wrapper for running Segment Anything 2 inference.

    This model supports:
      - image-mode propagation where `prompt_field` is a *sample-level*
        field
      - detection-mask prompts (when `Detection.mask` is present)
      - a monkey-patched `load_video_frames` implementation that can build
        SAM2's internal frame tensors from FiftyOne frame samples via a
        temporary symlink directory
    """

    def __init__(self, config):
        dir(sam2)  # ensure package is installed

        self._fields = {}

        self.config = config
        device = self.config.device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._download_model(config)

        try:
            self.ctx = _load_video_frames_monkey_patches()
        except Exception as e:
            logger.error(
                "Failed to monkey patch sam2.utils.misc.load_video_frames: %s",
                e,
            )
            self.ctx = None

        self.model = self._load_model(config)
        self.media_mode = getattr(config, "media_mode", "video")

        self._curr_prompt_type = None
        self._curr_prompts = None
        self._curr_negative_prompts = None
        self._curr_classes = None
        self._curr_frame_width = None
        self._curr_frame_height = None

    @property
    def media_type(self):
        return self.media_mode

    @property
    def ragged_batches(self):
        # Frames are resized to a fixed size, so batching is safe
        return False

    def _download_model(self, config):
        config.download_model_if_necessary()

    def _load_model(self, config):
        entrypoint = etau.get_function(config.entrypoint_fcn)
        if self.ctx is not None:
            with self.ctx:
                model = entrypoint(
                    config.entrypoint_args["model_cfg"],
                    ckpt_path=config.model_path,
                    device=self._device,
                )
        else:
            model = entrypoint(
                config.entrypoint_args["model_cfg"],
                ckpt_path=config.model_path,
                device=self._device,
            )
        return model

    def predict_all(
        self,
        imgs: List[np.ndarray],
        samples: Optional[List] = None,
    ) -> List[fol.Detections]:
        # Get the prompt field name from needs_fields (sample-level in this path)
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        if prompt_field is None:
            raise AttributeError(
                "Missing required argument 'prompt_field' for segment anything 2 video model"
            )

        # If there are no prompts anywhere in this sequence, do not call SAM2
        has_prompt = False
        for s in samples or []:
            val = s.get_field(prompt_field)
            if isinstance(val, fol.Detections):
                dets = cast(Iterable[fol.Detection], val.detections)
                if any(True for _ in dets):
                    has_prompt = True
                    break

        if not has_prompt:
            return [fol.Detections() for _ in (samples or [])]

        class _ImageSamplesAsVideoFrames:
            """Adapts a list of image samples to the video sample interface."""

            media_type = focm.IMAGE

            def __init__(self, frames):
                self._frames = list(frames)

            @property
            def frames(self):
                return {ii + 1: ff for ii, ff in enumerate(self._frames)}

        mock_video_sample = _ImageSamplesAsVideoFrames(samples)

        hh, ww = 0, 0
        if len(imgs) > 0:
            hh, ww = imgs[0].shape[0], imgs[0].shape[1]
        mock_reader = type("_Reader", (), {"frame_size": (ww, hh)})()

        # The existing video path will handle prompt extraction, registration,
        # propagation, and detection construction
        try:
            sample_detections = cast(
                dict[int, fol.Detections],
                self.predict(mock_reader, mock_video_sample),
            )
        except Exception as e:
            if "mat1 and mat2 must have the same dtype" in str(e):
                raise RuntimeError(
                    "Error propagating to all frames; please try with a shorter sequence with continuous frames and consistent labels."
                )
            raise

        return [
            sample_detections.get(i + 1, fol.Detections())
            for i in range(len(samples or []))
        ]

    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        if prompt_field is None:
            raise AttributeError(
                "Missing required argument 'prompt_field' for segment anything 2 video model"
            )

        # sample-level prompt_field for flattened frames
        if getattr(self, "media_mode", "video") == "video":
            if prompt_field.startswith("frames."):
                prompt_field = prompt_field[len("frames.") :]
            else:
                raise ValueError(
                    "'prompt_field' should be a frame field for segment anything 2 video model"
                )

        # Get negative_prompt_field if provided
        negative_prompt_field = None
        if "negative_prompt_field" in self.needs_fields:
            negative_prompt_field = self.needs_fields["negative_prompt_field"]
            if getattr(self, "media_mode", "video") == "video":
                if negative_prompt_field.startswith("frames."):
                    negative_prompt_field = negative_prompt_field[
                        len("frames.") :
                    ]
                else:
                    raise ValueError(
                        "'negative_prompt_field' should be a frame field for segment anything 2 video model"
                    )

        return prompt_field, negative_prompt_field

    def _get_prompt_type(self, sample, field_name):
        for _, frame in sample.frames.items():
            value = frame.get_field(field_name)
            if value is None:
                continue

            if isinstance(value, fol.Detections):
                detections = cast(Iterable[fol.Detection], value.detections)
                if any(det.mask is not None for det in detections):
                    return "masks"

                return "boxes"

            if isinstance(value, fol.Keypoints):
                return "points"

            raise ValueError(
                "Unsupported prompt type %s. The supported field types are %s"
                % (type(value), (fol.Detections, fol.Keypoints))
            )

        raise ValueError(
            f"Frame field {field_name} is empty for all frames, please provide at least one value"
        )

    def _get_prompts(self, sample, field_name):
        prompts = []
        for _, frame in sample.frames.items():
            value = frame.get_field(field_name)
            if value is not None:
                prompts.append(value)
            else:
                prompts.append([])

        return prompts

    def _forward_pass(self, video_reader, sample):
        if self._curr_prompt_type in ["boxes", "masks"]:
            return self._forward_pass_boxes(video_reader, sample)

        if self._curr_prompt_type == "points":
            return self._forward_pass_points(video_reader, sample)

        raise ValueError(f"Unsupported prompt_type {self._curr_prompt_type}")

    def _forward_pass_boxes(self, video_reader, sample):
        assert self._curr_prompts is not None
        assert self._curr_frame_width is not None
        assert self._curr_frame_height is not None

        image_folder = getattr(video_reader, "image_folder", None)
        if image_folder is not None:
            inference_state = self.model.init_state(image_folder)
        else:
            video_path = (sample, video_reader)
            inference_state = self.model.init_state(video_path)

        classes_obj_id_map = {}
        kp_idx_obj_id_map = {}
        current_obj_idx = 0

        for frame_idx, frame_detections in enumerate(self._curr_prompts):
            if (
                len(frame_detections) == 0
                or len(frame_detections.detections) == 0
            ):
                continue

            for detection in frame_detections.detections:
                if detection.index is not None:
                    if detection.index in kp_idx_obj_id_map:
                        ann_obj_id = kp_idx_obj_id_map[detection.index]
                    else:
                        ann_obj_id = current_obj_idx
                        kp_idx_obj_id_map[detection.index] = ann_obj_id
                        current_obj_idx += 1
                else:
                    ann_obj_id = current_obj_idx
                    current_obj_idx += 1

                classes_obj_id_map[ann_obj_id] = detection.label

                box_xyxy = fosam._to_abs_boxes(
                    np.array([detection.bounding_box]),
                    self._curr_frame_width,
                    self._curr_frame_height,
                    chunk_size=1,
                )
                box = np.round(box_xyxy.squeeze(axis=0)).astype(int)

                if detection.mask is not None:
                    # Prevent SAM2 from running a segmentation inside the box.
                    # Instead, use the prompted mask directly.
                    mask_array = _to_abs_mask(
                        detection.mask,
                        box,
                        self._curr_frame_width,
                        self._curr_frame_height,
                    )
                    _, _, _ = self.model.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=ann_obj_id,
                        mask=mask_array,
                    )
                else:
                    _, _, _ = self.model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=ann_obj_id,
                        box=box,
                    )

        sample_detections = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.model.propagate_in_video(inference_state):
            detections = []

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = np.squeeze(
                    (out_mask_logits[i] > 0.0).cpu().numpy(), axis=0
                )

                if self._curr_negative_prompts and out_frame_idx < len(
                    self._curr_negative_prompts
                ):
                    mask = _subtract_negative_box_regions(
                        mask,
                        self._curr_negative_prompts[out_frame_idx],
                        self._curr_frame_width,
                        self._curr_frame_height,
                    )

                box = fosam._mask_to_box(mask)
                if box is None:
                    continue

                label = classes_obj_id_map[out_obj_id]
                x1, y1, x2, y2 = box

                bounding_box = [
                    x1 / self._curr_frame_width,
                    y1 / self._curr_frame_height,
                    (x2 - x1) / self._curr_frame_width,
                    (y2 - y1) / self._curr_frame_height,
                ]

                mask = mask[
                    int(round(y1)) : int(round(y2)),
                    int(round(x1)) : int(round(x2)),
                ]

                detections.append(
                    fol.Detection(
                        label=label,
                        bounding_box=bounding_box,
                        mask=mask
                        if self._curr_prompt_type == "masks"
                        else None,
                        index=out_obj_id,
                    )
                )

            sample_detections[int(out_frame_idx) + 1] = fol.Detections(
                detections=detections
            )

        return sample_detections

    def _forward_pass_points(self, video_reader, sample):
        assert self._curr_prompts is not None
        assert self._curr_frame_width is not None
        assert self._curr_frame_height is not None

        image_folder = getattr(video_reader, "image_folder", None)
        if image_folder is not None:
            inference_state = self.model.init_state(image_folder)
        else:
            video_path = (sample, video_reader)
            inference_state = self.model.init_state(video_path)

        classes_obj_id_map = {}
        kp_idx_obj_id_map = {}
        current_obj_idx = 0

        for frame_idx, frame_keypoints in enumerate(self._curr_prompts):
            if (
                len(frame_keypoints) == 0
                or len(frame_keypoints.keypoints) == 0
            ):
                continue

            for keypoint in frame_keypoints.keypoints:
                if keypoint.index is not None:
                    if keypoint.index in kp_idx_obj_id_map:
                        ann_obj_id = kp_idx_obj_id_map[keypoint.index]
                    else:
                        ann_obj_id = current_obj_idx
                        kp_idx_obj_id_map[keypoint.index] = ann_obj_id
                        current_obj_idx += 1
                else:
                    ann_obj_id = current_obj_idx
                    current_obj_idx += 1

                classes_obj_id_map[ann_obj_id] = keypoint.label

                points, labels = fosam._to_sam_points(
                    keypoint.points,
                    self._curr_frame_width,
                    self._curr_frame_height,
                    keypoint,
                )

                if self._curr_negative_prompts and frame_idx < len(
                    self._curr_negative_prompts
                ):
                    neg_frame_keypoints = self._curr_negative_prompts[
                        frame_idx
                    ]
                    if (
                        neg_frame_keypoints
                        and isinstance(neg_frame_keypoints, fol.Keypoints)
                        and len(neg_frame_keypoints.keypoints) > 0
                    ):
                        neg_keypoints = cast(
                            Iterable[Any], neg_frame_keypoints.keypoints
                        )
                        for neg_keypoint in neg_keypoints:
                            neg_points, _ = fosam._to_sam_points(
                                neg_keypoint.points,
                                self._curr_frame_width,
                                self._curr_frame_height,
                                neg_keypoint,
                            )
                            neg_labels = np.zeros(len(neg_points), dtype=int)
                            points = np.vstack([points, neg_points])
                            labels = np.concatenate([labels, neg_labels])

                _, _, _ = self.model.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )

        sample_detections = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.model.propagate_in_video(inference_state):
            detections = []

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = np.squeeze(
                    (out_mask_logits[i] > 0.0).cpu().numpy(), axis=0
                )

                box = fosam._mask_to_box(mask)
                if box is None:
                    continue

                label = classes_obj_id_map[out_obj_id]
                x1, y1, x2, y2 = box

                bounding_box = [
                    x1 / self._curr_frame_width,
                    y1 / self._curr_frame_height,
                    (x2 - x1) / self._curr_frame_width,
                    (y2 - y1) / self._curr_frame_height,
                ]

                mask = mask[
                    int(round(y1)) : int(round(y2)),
                    int(round(x1)) : int(round(x2)),
                ]

                detections.append(
                    fol.Detection(
                        label=label,
                        bounding_box=bounding_box,
                        mask=mask,
                        index=out_obj_id,
                    )
                )

            sample_detections[int(out_frame_idx) + 1] = fol.Detections(
                detections=detections
            )

        return sample_detections


def load_fiftyone_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=None,
):
    """
    The signature of this function matches
    `sam2.utils.misc.load_video_frames`.

    Note:
        The argument `video_path` is a misnomer; we pass a tuple of
        (sample, reader) instead.
    """
    sample, reader = video_path

    if sample.media_type == focm.VIDEO:
        return load_fiftyone_video_frames_from_video_file(
            sample=sample,
            video_reader=reader,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device=compute_device,
        )

    if sample.media_type == focm.IMAGE:
        return load_fiftyone_video_frames_from_image_files(
            sample=sample,
            image_size=image_size,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=False,
        )

    raise NotImplementedError("Unsupported media type")


def load_fiftyone_video_frames_from_video_file(
    sample,
    video_reader,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=None,
):
    if compute_device is None:
        compute_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    num_frames = len(sample.frames)
    images = torch.zeros(
        num_frames, 3, image_size, image_size, dtype=torch.float32
    )

    for frame_number in range(num_frames):
        current_frame = video_reader.read()
        resized_frame = (
            cv2.resize(current_frame, (image_size, image_size)) / 255.0
        )
        img = torch.from_numpy(resized_frame).permute(2, 0, 1)
        images[frame_number] = img

    video_width, video_height = (
        current_frame.shape[1],
        current_frame.shape[0],
    )

    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def load_fiftyone_video_frames_from_image_files(
    sample,
    image_size,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
):
    """Load video frames from FiftyOne frame samples.

    Creates a temp dir, symlinks each frame filepath as 00000.jpg, 00001.jpg,
    ..., (sorted by frame number), then calls SAM2's original
    `load_video_frames_from_jpg_images` function for that path.

    Temp dir is removed on return.
    """
    with tempfile.TemporaryDirectory(prefix="fo_sam2_frames_") as tmpdir:
        frame_filepaths = [
            get_local_path(sample.frames[ii])
            for ii in sorted(sample.frames.keys())
        ]

        for idx, frame_filepath in enumerate(frame_filepaths):
            dest = os.path.join(tmpdir, "%05d.jpg" % idx)
            os.symlink(os.path.abspath(frame_filepath), dest)

        return smutil.load_video_frames_from_jpg_images(
            tmpdir,
            image_size=image_size,
            offload_video_to_cpu=True,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
        )


def _load_video_frames_monkey_patches():
    return fou.MonkeyPatchFunction(
        smutil.load_video_frames, load_fiftyone_video_frames
    )
