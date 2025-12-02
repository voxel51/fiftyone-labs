"""Video decoding and dataloading for FiftyOne model inference."""

import contextlib
import numpy as np
from collections import OrderedDict

import eta.core.video as etav
import fiftyone as fo
import fiftyone.core.media as fom
import fiftyone.core.utils as fou
from fiftyone.core.models import (
    ErrorHandlingCollate,
    TorchModelMixin,
)
import fiftyone.utils.torch as fout

torch = fou.lazy_import("torch")
tud = fou.lazy_import("torch.utils.data")
tcd = fou.lazy_import("torchcodec.decoders")

# Video decoder cache
_worker_decoder_cache = None


def apply_image_model_to_video_frames(
    samples: fo.core.collections.SampleCollection,
    model: fo.core.models.Model,
    label_field="predictions",
    confidence_thresh=None,
    batch_size=None,
    frames_chunk_size=None,
    parse_type="sequential",
    num_workers=None,
    skip_failures=True,
    progress=None,
):
    """Applies the image model to the video samples in the collection.

    Only supports applying image model to videos frames.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        model: a :class:`Model`for frame-by-frame video inference
        label_field ("predictions"): the name of the field in which to store
            the model predictions. When performing inference on video frames,
            the "frames." prefix is optional
        confidence_thresh (None): Confidence threshold for filtering labels
        batch_size (None): number of videos to process as a batch
        frames_chunk_size (None): number of frames in one chunk
        parse_type ("interleaved"): whether to parse video chunks in an interleaved
            manner or read all chunks from one video before moving to the next video
        num_workers (None): the number of workers to use when loading videos
        skip_failures (True): whether to gracefully continue without raising an
            error if predictions cannot be generated for a sample. Only
            applicable to :class:`Model` instances
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead

    """
    if samples.media_type != fom.VIDEO:
        raise fom.MediaTypeError(
            f"Unsupported media type {samples.media_type}."
        )

    if model.media_type != "image":
        raise ValueError("Only image models are supported.")

    label_field, _ = samples._handle_frame_field(label_field)

    data_loader = _make_video_frame_data_loader(
        samples=samples,
        model=model,
        batch_size=batch_size,
        parse_type=parse_type,
        num_workers=num_workers,
        frames_chunk_size=frames_chunk_size,
        skip_failures=skip_failures,
    )

    with contextlib.ExitStack() as context:
        if confidence_thresh is not None and hasattr(
            model.config, "confidence_thresh"
        ):
            context.enter_context(
                fou.SetAttributes(
                    model.config, confidence_thresh=confidence_thresh
                )
            )
            confidence_thresh = None

        pb = context.enter_context(
            fou.ProgressBar(data_loader, progress=progress)
        )
        context.enter_context(fou.SetAttributes(model, preprocess=False))

        for batch in pb(data_loader):
            for frames in batch:
                labels_frames = model.predict_all(frames["frames"])
                sample_idx = frames["sample_idx"]
                sample = samples[sample_idx]

                fns = frames["frame_ids"]
                sample.add_labels(
                    {
                        int(fn): labels
                        for fn, labels in zip(fns, labels_frames)
                    },
                    label_field=label_field,
                    confidence_thresh=confidence_thresh,
                )
                sample.save()


def worker_init_fn(worker_id):
    global _worker_decoder_cache, _worker_iterator_cache
    _worker_decoder_cache = OrderedDict()
    _worker_iterator_cache = OrderedDict()


def _get_cached_decoder(video_path, max_cached):
    global _worker_decoder_cache, _worker_iterator_cache

    if _worker_decoder_cache is None:
        _worker_decoder_cache = OrderedDict()
        _worker_iterator_cache = OrderedDict()

    if video_path in _worker_decoder_cache:
        _worker_decoder_cache.move_to_end(video_path)
        _worker_iterator_cache.move_to_end(video_path)
        return (
            _worker_decoder_cache[video_path],
            _worker_iterator_cache[video_path],
        )

    decoder = etav.FFmpegVideoReader(video_path)
    iterator = iter(decoder)
    _worker_decoder_cache[video_path] = decoder
    _worker_iterator_cache[video_path] = iterator

    if len(_worker_decoder_cache) > max_cached:
        _del_lru_decoder()

    return decoder, iterator


def _del_lru_decoder():
    global _worker_decoder_cache, _worker_iterator_cache

    if not _worker_decoder_cache:
        return

    old_path, old_decoder = _worker_decoder_cache.popitem(last=False)
    old_decoder.close()
    del _worker_iterator_cache[old_path]


def _make_video_frame_data_loader(
    samples,
    model,
    batch_size,
    num_workers,
    frames_chunk_size,
    parse_type,
    skip_failures,
):
    use_numpy = not isinstance(model, TorchModelMixin)
    num_workers = fout.recommend_num_workers(num_workers)

    if batch_size is None:
        batch_size = 1

    collate_fn = ErrorHandlingCollate(
        skip_failures,
        ragged_batches=model.ragged_batches,
        use_numpy=use_numpy,
        user_collate_fn=_make_video_collate(model.collate_fn),
    )

    dataset = TorchVideoFramesDataset(
        samples=samples,
        transform=model.transforms,
        include_ids=True,
        chunk_size=frames_chunk_size,
        parse_type=parse_type,
        skip_failures=skip_failures,
    )
    pin_memory = isinstance(model, fout.TorchImageModel) and model._using_gpu

    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )


def _make_video_collate(model_collate):
    def _video_collate_fn(batch):
        for b in batch:
            frames_data = b["frames"]
            collated_frames = model_collate(frames_data)
            b["frames"] = collated_frames
        return batch

    return _video_collate_fn


class TorchVideoFramesDataset(tud.Dataset):
    def __init__(
        self,
        video_paths=None,
        samples=None,
        sample_ids=None,
        include_ids=False,
        transform=None,
        chunk_size=None,
        parse_type="sequential",
        skip_failures=False,
        max_cached_decoders=8,
    ):
        self.chunk_size = chunk_size if chunk_size is not None else 1
        self.parse_type = parse_type
        self.max_cached_decoders = max_cached_decoders

        video_paths, sample_ids, chunk_index = self._parse_inputs(
            video_paths=video_paths,
            samples=samples,
            sample_ids=sample_ids,
            include_ids=include_ids,
        )

        self.video_paths = video_paths
        self.sample_ids = sample_ids
        self.chunk_index = chunk_index
        self.transform = transform
        self.skip_failures = skip_failures

    def __len__(self):
        return len(self.chunk_index)

    def _parse_inputs(
        self,
        video_paths=None,
        samples=None,
        sample_ids=None,
        include_ids=False,
    ):
        if video_paths is None and samples is None:
            raise ValueError(
                "Either `video_paths` or `samples` must be provided"
            )

        if video_paths is None:
            video_paths = samples.values("filepath")

        if include_ids and sample_ids is None:
            sample_ids = samples.values("id")

        samples_metadata = None
        if samples is not None and samples.count("metadata") < len(samples):
            samples.compute_metadata()
        samples_metadata = samples.values("metadata")

        chunk_index = []
        vid_to_chunks = []
        vid_to_frame_ids = []
        for idx, video_path in enumerate(video_paths):
            if samples_metadata:
                total_frames = samples_metadata[idx].total_frame_count
            else:
                raise Exception(f"No metadata: {video_path}")

            # Process all frames in chunks
            frame_ids = np.arange(0, total_frames)
            vid_to_chunks.append(len(frame_ids) // self.chunk_size)
            vid_to_frame_ids.append(frame_ids)

        # Create chunk index
        if self.parse_type == "sequential":
            for idx, frame_ids in enumerate(vid_to_frame_ids):
                for i in range(0, len(frame_ids), self.chunk_size):
                    chunk_index.append(
                        (idx, frame_ids[i : i + self.chunk_size])
                    )
        elif self.parse_type == "interleaved":
            max_chunks = max(v for v in vid_to_chunks)
            for round_idx in range(max_chunks):
                for idx, video_chunks in enumerate(vid_to_chunks):
                    if round_idx < video_chunks:
                        frame_ids = vid_to_frame_ids[idx]
                        start = round_idx * self.chunk_size
                        end = min(start + self.chunk_size, len(frame_ids))

                        chunk_index.append((idx, frame_ids[start:end]))
        else:
            raise Exception("Invalid parsing type for chunks")

        return video_paths, sample_ids, chunk_index

    def __getitem__(self, idx):
        vid_idx, frame_ids = self.chunk_index[idx]
        video_path = self.video_paths[vid_idx]
        _, iterator = _get_cached_decoder(video_path, self.max_cached_decoders)

        chunk_size = len(frame_ids)
        frames = []
        for _ in range(chunk_size):
            try:
                img = next(iterator)
                frames.append(np.array(img))
            except StopIteration:
                break

        transformed_frames = []
        if self.transform:
            for frame in frames:
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)

        sample_id = self.sample_ids[vid_idx] if self.sample_ids else None
        frames_data = transformed_frames if transformed_frames else frames

        return {
            "frames": frames_data,
            "sample_idx": sample_id,
            "video_path": video_path,
            "frame_ids": frame_ids + 1,
        }
