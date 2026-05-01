# Label Propagation Benchmarking Plan

## Background

The plugin uses SAM-2 to propagate sparse annotations (e.g., detections on frame 0 of each sequence) to all subsequent frames. The stated performance target is **< 100ms per frame** (currently unmet). Key knobs: `batch_size`, device (CPU vs GPU), number of sequences, and frames per sequence.

---

## What to Measure

For each experiment, record:

| Metric | How to get it |
|--------|---------------|
| **Total wall time** | `time.time()` around the operator call |
| **Frames/sec** | `total_frames / total_wall_time` |
| **Time per sequence** | wall time divided by num sequences |
| **GPU memory peak** | `nvidia-smi --query-gpu=memory.used --format=csv -l 1` in a side terminal |
| **GPU utilization %** | same `nvidia-smi` loop |
| **CPU utilization %** | `htop` or `psutil` during run |
| **RAM used (GB)** | `psutil.Process().memory_info().rss / 1e9` |

A minimal timing wrapper to add around the operator call in your script:

```python
import time, psutil, os

t0 = time.perf_counter()
# --- run propagation operator here ---
t1 = time.perf_counter()

elapsed = t1 - t0
n_frames = sum(len(seq) for seq in sequences)   # adjust to your data shape
print(f"Elapsed: {elapsed:.1f}s | Frames: {n_frames} | ms/frame: {1000*elapsed/n_frames:.1f}")
```

---

## Experiment Grid

### Axis 1 — Device (CPU vs GPU)

`batch_size=32`, 5 MOSE sequences run independently (pytest parametrized). Sequences vary in length (3–120 frames).

#### GPU (cuda:0)

`reset_peak_memory_stats()` is called after the model loads, so GPU VRAM delta ≈ GPU VRAM peak (model weights are already in VRAM before the timer starts).

| Seq | Frames | Total time (s) | ms/frame | RAM total (GB) | RAM delta (GB) | GPU VRAM delta (GB) |
|-----|--------|---------------|----------|---------------|----------------|---------------------|
| 1   | 36     | 11.55         | 320.97   | 1.51          | +0.29          | 0.84                |
| 2   | 120    | 20.31         | 169.26   | 1.82          | +0.38          | 0.84                |
| 3   | 45     | 13.80         | 306.75   | 1.73          | −0.09          | 0.90                |
| 4   | 65     | 11.99         | 184.46   | 1.89          | +0.22          | 0.83                |
| 5   | 3      | 1.95          | 648.56   | 1.86          | +0.09          | 0.76                |
| **Total** | **269** | **59.60** | **221.6** | — | — | **0.90 max** |

#### CPU

GPU VRAM delta not meaningful for CPU runs (no CUDA compute; residual ~0.11–0.14 GB is CUDA context overhead).

| Seq | Frames | Total time (s) | ms/frame | RAM total (GB) | RAM delta (GB) |
|-----|--------|---------------|----------|---------------|----------------|
| 1   | 36     | 59.48         | 1652.26  | 1.96          | +0.67          |
| 2   | 120    | 197.47        | 1645.57  | 2.67          | +0.72          |
| 3   | 45     | 153.25        | 3405.55  | 2.97          | +0.54          |
| 4   | 65     | 113.94        | 1752.87  | 2.90          | +0.38          |
| 5   | 3      | 6.32          | 2105.07  | 2.77          | +0.09          |
| **Total** | **269** | **530.46** | **1972** | — | — |

**GPU speedup: ~8.9×** (530.46s vs 59.60s). Well within the expected 5–20× range.

**Notable observations:**
- GPU VRAM is tiny — 0.90 GB peak with SAM-2 tiny. No VRAM pressure at any batch size on a 45 GB card.
- CPU seq 3 is an outlier at 3406 ms/frame vs ~1650 for the others — worth a second look (shared machine contention? different sequence characteristics?).
- Short sequences (seq 5, 3 frames) show inflated ms/frame on both devices — startup/model-init overhead dominates. Exclude from throughput conclusions.
- The 100 ms/frame target is **not met** on GPU at batch=32. Axis 2 (batch size sweep) will show whether larger batches close the gap.

---

### Axis 2 — Batch Size (GPU cuda:4, 5 MOSE sequences)

Rows = batch size. Per-sequence columns show ms/frame (frame count fixed per sequence, shown in header). Seq 5 (3 frames) is startup-dominated and excluded from the weighted average.

GPU VRAM delta unreliable for this axis (model was on cuda:4, stats queried from cuda:0 — see Axis 2 note above).

| Batch | Seq1 (36f) ms/f | Seq2 (120f) ms/f | Seq3 (45f) ms/f | Seq4 (65f) ms/f | Seq5 (3f) ms/f | Weighted avg ms/f† | Avg RAM total (GB) |
|-------|----------------|-----------------|----------------|----------------|---------------|-------------------|--------------------|
| 4     | 352.3          | 204.1           | 274.1          | 217.5          | 2002.9        | **239.2**         | 1.39               |
| **8** | **354.7**      | **183.3**       | **287.1**      | **217.3**      | **1984.1**    | **232.4**         | **1.41**           |
| 16    | 424.3          | 194.9           | 343.7          | 226.1          | 2005.0        | 258.8             | 1.66               |
| 32    | 450.1          | 191.3           | 410.5          | 242.3          | 2136.2        | 275.9             | 1.97               |
| 64    | 500.6          | 231.9           | 404.6          | 231.6          | 1965.4        | 297.4             | 2.03               |

†Weighted avg = total time across seqs 1–4 / total frames across seqs 1–4.

**Observations:**
- **batch=8 is the sweet spot** — best weighted avg ms/frame (232 ms/f) with minimal RAM (1.41 GB).
- Performance degrades monotonically beyond batch=8. Counterintuitively, larger batches are *slower* despite having fewer chunk boundaries (and thus fewer overlap frames). Cause unclear — possibly GPU memory transfer overhead for larger frame buffers, or per-`apply_model`-call overhead that's worse for fewer, larger calls. Worth investigating with longer sequences (Axis 3) to see if the trend holds.
- RAM grows with batch size (1.4 → 2.0 GB) but stays low in absolute terms for SAM-2 tiny.
- The 100 ms/frame target remains unmet even at the best batch size.

---

### Axis 3 — Dataset Scale (GPU cuda:4, batch=8 and batch=32)

Each row = a run over N sequences (same pytest parametrized setup, sequences accumulated sequentially). "Final RAM" = process RSS at the end of the last sequence — represents peak RAM for the whole run.

GPU VRAM delta unreliable for this axis (same device mismatch as Axis 2).

| # Seqs | Total frames | bs=8 total time (s) | bs=8 ms/f† | bs=8 final RAM (GB) | bs=32 total time (s) | bs=32 ms/f† | bs=32 final RAM (GB) |
|--------|-------------|--------------------|-----------:|--------------------:|---------------------|-----------:|---------------------:|
| 5      | 269         | 67.8               | 232.4      | 1.48                | 79.8                | 275.9      | 2.11                 |
| 10     | 640         | 136.0              | 194.9      | 1.88                | 144.3               | 208.7      | 2.91                 |
| 20     | 1,331       | 260.6              | 187.3      | 2.26                | 277.6               | 200.1      | 3.44                 |
| 50     | 3,801       | 717.6              | 184.3      | 2.90                | 745.3               | 191.6      | 5.58                 |

†Weighted avg ms/frame excluding sequences with ≤6 frames (startup-dominated).

**Observations:**
- **Scaling is linear** — time/sequence stays ~13–14s (bs=8) and ~14–15s (bs=32) across all scales. No quadratic growth or memory pressure degradation.
- **ms/frame improves as N grows** — not because SAM-2 gets faster, but because the sequence pool adds longer sequences that amortize per-sequence startup cost better.
- **RAM growth is the main concern at scale:** bs=32 grows from 1.5 → 5.6 GB over 50 sequences (FiftyOne caching sample metadata), vs bs=8 which stays under 3 GB. On a shared machine, bs=32 at 50+ sequences needs monitoring.
- **bs=8 wins consistently** at every scale, confirming Axis 2 finding.
- **100 ms/frame target remains unmet** at all scales.

---

### Axis 4 — Video vs Image Groups (DAVIS, GPU cuda:4, 5 sequences)

Same 5 sequences (69, 50, 80, 84, 90 frames), run as image groups vs native video. GPU peak values here are valid (device fix was applied).

GPU VRAM delta valid here (device fix applied). Same reasoning as Axis 1: `reset_peak_memory_stats()` fires after model load, so reported peak ≈ inference delta.

| Dataset type | Seq | Frames | Time (s) | ms/frame | GPU VRAM delta (GB) | RAM total (GB) |
|-------------|-----|--------|---------|---------|---------------------|---------------|
| Image groups | 1   | 69     | 8.43    | 122.1   | 0.87                | 1.46          |
| Image groups | 2   | 50     | 4.26    | 85.1    | 0.79                | 1.57          |
| Image groups | 3   | 80     | 9.25    | 115.6   | 0.89                | 1.71          |
| Image groups | 4   | 84     | 6.73    | 80.2    | 0.82                | 1.79          |
| Image groups | 5   | 90     | 7.49    | 83.3    | 0.82                | 1.97          |
| **Image total** | — | **373** | **36.2** | **96.9** | **0.84 avg** | —    |
| Video        | 1   | 69     | 6.93    | 100.4   | 1.71                | 1.41          |
| Video        | 2   | 50     | 3.20    | 64.0    | 1.39                | 1.42          |
| Video        | 3   | 80     | 8.05    | 100.6   | 1.87                | 1.52          |
| Video        | 4   | 84     | 5.61    | 66.8    | 1.85                | 1.29          |
| Video        | 5   | 90     | 5.62    | 62.4    | 1.93                | 1.34          |
| **Video total** | — | **373** | **29.4** | **78.8** | **1.75 avg** | —   |

**Observations:**
- **Video mode is ~19% faster** (78.8 vs 96.9 ms/frame overall).
- **Video mode meets the <100ms/frame target** — all but two sequences are under 100ms/f, and the weighted avg is 78.8ms/f. Image groups just miss it (96.9ms/f avg, seqs 1 and 3 go over 100ms/f).
- **Video mode uses ~2× the VRAM** (1.75 GB avg peak vs 0.84 GB) — it processes the full video in one pass without chunking, holding more in memory simultaneously. Still trivial on a 45 GB card.
- The speedup is from skipping the image-mode chunking overhead (overlap frame re-processing, extra DB reads/writes). Video mode hands the full sequence to SAM-2 in one shot.

---

## MOSE Sample Size Guidance

MOSE sequences average ~100–200 frames each (the train split has longer clips than val).

| Phase | # Sequences | Reasoning |
|-------|------------|-----------|
| Smoke test / correctness | 1–2 | Fast iteration; verify output fields are correct before timing anything |
| Per-axis sweep (Axes 1–2) | 5 | Enough frames (~750) to amortize startup cost; matches the existing demo |
| Scale axis (Axis 3) | 1 → 50 | Go up in doublings; stop when you hit OOM or patience runs out |
| Final "production" run | 20–50 | Representative of real workloads |

The demo script (`run_demo_mose.py`) uses indices `[4, 6, 8, 10, 12]` — 5 sequences. For larger sweeps, just change the slice length, e.g.:

```python
# in run_demo_mose.py — change this line:
scene_indices = list(range(num_scenes))          # sequential
# or random sample for representativeness:
import random; scene_indices = random.sample(range(total_scenes), num_scenes)
```

---

## Instrumentation Checklist

Before starting:
- [ ] Confirm `nvidia-smi` works on the remote machine and shows both GPUs
- [ ] Verify SAM-2 weights are already downloaded (first run downloads ~100 MB, skewing Exp S1/D1)
- [ ] Pin `torch` and `segment-anything-2` versions in a note — results are only comparable within the same environment
- [ ] Record GPU model (e.g., A100 80GB vs RTX 3090 24GB) — batch size limits depend on VRAM

During each run:
```bash
# in a second terminal, log GPU stats every 2 seconds to a file
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free \
           --format=csv -l 2 > gpu_log_expXX.csv
```

After each run, note in the table:
- Did any sequences fail / throw errors?
- Were there any CUDA OOM warnings (might silently fall back)?
- Was the output field populated for all frames?

---

## Questions This Should Answer

1. **Is GPU worth it?** (Axis 1 — if CPU is already fast enough, simpler deployment)
2. **What batch size to recommend in docs?** (Axis 2 — the knee point)
3. **Does it scale linearly?** (Axis 3 — important for large datasets)
4. **How long will a "real" job take?** (Axis 3, S5–S6 → extrapolate to full MOSE train split)
5. **Is the 100ms/frame target achievable on GPU?** (check ms/frame column in D2)
