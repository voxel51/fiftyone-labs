# Label propagation evaluation metrics

This note describes the **primary** geometric metric implemented as `evaluate()` in `suc_utils.py`, how it differs from COCO-style detection evaluation, and how it relates to **`evaluate_success_rate()`**. Implementation details live in code; this file is the conceptual summary.

---

## `evaluate()` — Hungarian mean IoU

**Purpose.** Score how well a propagated set of boxes aligns with a reference set (typically ground truth vs propagated predictions) on **one frame** (or one sample), without requiring detection confidences or a fixed IoU threshold.

**Pipeline (conceptually).**

1. **Drop degenerate boxes** — Detections with zero bounding-box area (`width × height == 0` in normalized coordinates) are removed. They are treated as non-participating placeholders so they neither inflate nor deflate the score.

2. **Cardinality** — Let \(G\) be the number of remaining reference (“ground truth”) boxes and \(P\) the number of remaining predicted boxes. If both are zero, the score is **1.0** (nothing to score). If exactly one side is empty, the score is **0.0** (pure miss or pure hallucination).

3. **Pairwise IoU** — For every reference box \(i\) and every predicted box \(j\), compute axis-aligned bounding-box **IoU** in normalized \([0,1]^2\) space.

4. **Optimal matching** — Build an IoU matrix of shape \(G \times P\) and solve a **linear assignment** problem that **maximizes** the sum of IoUs of chosen pairs (Hungarian algorithm on costs \(1 - \text{IoU}\), equivalently).

5. **Scalar score** — Sum the IoUs of all matched pairs, then divide by **\(\max(G, P)\)**. Unmatched boxes contribute **implicitly as zero** in the numerator, but still count in the denominator through the larger cardinality.

**Reading the number.** The result is in **\([0, 1]\)**. It is **not** a calibrated probability. It is best interpreted as a **smooth localization quality** score: perfect overlap for all boxes in the larger set yields **1.0**; disjoint sets trend toward **0.0**; partial overlap gives intermediate values without committing to “pass/fail” at a single IoU cutoff.

**Why \(\max(G, P)\)?** Extra predictions or missing references both **lower** the score because the same matched IoU mass is spread across more “slots.” That matches the intuition that propagation should neither drop objects nor invent spurious boxes, while still rewarding good overlap for the matches that exist.

**Contrast with COCO-style eval.** Metrics such as precision / recall / F1 at a fixed IoU (e.g. FiftyOne’s `evaluate_detections`) mark each prediction as TP / FP and each missed GT as FN. They are ideal for **benchmark-style** reporting at an operating point. **`evaluate()`** is better suited to **per-frame grading** and optimization feedback: it varies continuously with small box shifts and does not require confidence scores for a single scalar summary.

---

## `evaluate_success_rate()` — area under a success curve

**Definition (informally).** After the same preprocessing and **same Hungarian matching** as `evaluate()`, collect the list of per-entity scores: matched IoUs plus **zero** for each unmatched reference or prediction (with only positive-area boxes in play after dropping degenerates). Sort those values, then integrate a **success-vs-threshold** curve: at each distinct IoU threshold, ask what fraction of entries are at least that high, and accumulate a weighted area (see code for the exact discrete construction).

**Equivalence to `evaluate()`.** For this construction, that curve integral collapses to the **arithmetic mean** of the multiset above, which equals **sum of matched IoUs / \(\max(G, P)\)\`** — i.e. the same value as **`evaluate()`**. So the two functions return the same scalar under the shared matching and degenerate-box rules; **`evaluate_success_rate()`** is kept mainly for historical / plotting alignment with “success plot” language, not as the main API to highlight, because the implementation path is more involved for an identical number.

---

## Related: `evaluate_matched()`

When object identity is **fixed by list order** (same length, \(k\)-th prediction compared to \(k\)-th reference), use **`evaluate_matched()`**: mean IoU over aligned pairs, skipping pairs where **both** sides are degenerate placeholders. That is the right shape for **slot-aligned** tracks; **`evaluate()`** is the right shape for **set** alignment when counts may differ.

---

## Limitations (current design)

- **Symmetry in cardinality, not in value.** Swapping “reference” and “propagated” does not change the score when both sides use the same preprocessing, but **`evaluate()`** does not by itself separate “too many predictions” from “too few”; both appear as a lower scalar. Metrics that split precision and recall remain useful when you need that asymmetry.

- **Bounding boxes only.** IoU is computed on normalized rectangles. Mask geometry is not used in these helpers yet.

- **No confidence / scores.** Predictions are not weighted by model confidence; every remaining box counts equally in the assignment problem.

- **Labels and indices.** Matching is purely geometric today. Restricting the IoU matrix to pairs that share a **label** and/or an **`index`** (when you add that) aligns the metric with **instance identity**, analogous to how COCO-style matching respects class labels.

---

## When to use what

| Goal                                                      | Suggestion                                             |
| --------------------------------------------------------- | ------------------------------------------------------ |
| Smooth per-frame quality for propagation / tracking debug | **`evaluate()`** (or matched variant when slots align) |
| Agreement with common detection benchmarks at a fixed IoU | **COCO-style** (e.g. FiftyOne `evaluate_detections`)   |
