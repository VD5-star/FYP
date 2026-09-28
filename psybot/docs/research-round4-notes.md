# Round 4 — measured findings and their limits

Work done while the user was away. Every number here was produced by running
code on this machine; every limitation is stated rather than omitted.

---

## Finding 1 — the synthetic clip cannot measure tracking under motion

This invalidates part of Round 2 and must be recorded before anything built on
it is trusted.

The `squat.mp4` clip animates a figure through a known knee-bend cycle. The
ground-truth hip height is known exactly, so tracking quality is directly
checkable. It was checked:

| Signal | Result |
|---|---|
| Detected hip-height range | 120 px |
| Ground-truth hip-height range | 115 px |
| **Correlation, detected hip height against the driving knee-bend signal** | **−0.04** |

A correlation of −0.04 across every tested lag means **MediaPipe is not tracking
the squat at all.** It produces hip positions that vary over a plausible range
while being uncorrelated with what the figure is actually doing.

Per-frame offsets confirm it: −164 px at frame 20, −32 px at frame 80, −109 px
at frame 120. Not a constant bias that could be calibrated away — it wanders.

### Why

The figure is drawn with OpenCV primitives: flat polygons, uniform colour, hard
edges, no texture, no shading, no clothing folds. MediaPipe is trained on
photographs. It finds *something* body-shaped and reports high confidence for
the torso, but the per-frame landmark positions are not locked to the rendered
geometry.

This is the same failure that made `heavy` look sixteen times worse in Round 2 —
and it is now clear the problem is broader than that one model.

### Consequence

| Measurement | Still valid? |
|---|---|
| Inference latency (11.8 / 17.5 / 57.3 ms) | **Yes** — timing does not depend on what is in the image |
| Detection rate | Partly — it shows the figure is *found*, not that it is *tracked* |
| Joint-angle noise floor on `still.mp4` | **Questionable** — a still clip has nothing to track, so this measures noise on a fixed input, which is still meaningful, but the absolute value will not transfer |
| Knee-angle error "2.6°" from Round 2 | **Withdrawn.** Measured on 58 of 290 frames against a figure the model was not actually tracking |
| Anything about motion tracking | **No** |

**The 2.6° figure must not appear in the report.** Round 2 already flagged the
sample as too small; this round shows it was measuring nothing.

### What this does not invalidate

The latency measurements stand, and with them the central Round 2 conclusion:
**the 14.8 fps observed live was never model-bound.** Inference is 17.5 ms
regardless of image content.

---

## Finding 2 — our extra smoothing filter buys almost nothing

Measured on `still.mp4`, left-wrist pixel jitter (true variance is zero, so all
of it is error):

| Configuration | Jitter |
|---|---|
| MediaPipe only, no extra filter | 0.837 px |
| **Ours (`min_cutoff` 1.2, `beta` 0.02)** | **0.702 px** |
| Gentler (0.5, 0.05) | 0.702 px |
| Near-MediaPipe's own constants (0.05, 80) | 0.839 px |

**Our `PoseSmoother` reduces jitter by 16%.** Not nothing, but far less than its
presence implies — because, as Round 3 discovered, MediaPipe already applies a
One-Euro filter internally with published constants:

```
screen landmarks : min_cutoff 0.05, beta 80.0
visibility       : low_pass alpha 0.1
```

We are filtering an already-filtered signal.

Two further observations:

1. Tuning our filter gentler (0.5, 0.05) gives **identical** jitter to our
   current setting. The parameter is doing less than it appears.
2. Setting it to MediaPipe's own constants makes jitter slightly **worse** than
   no filter at all (0.839 vs 0.837) — consistent with double-filtering adding
   phase distortion without adding suppression.

### The measurement that could not be completed

The real question is the trade: 16% less jitter in exchange for how much lag?

Lag was to be measured as the cross-correlation delay between the detected hip
height and the ground truth during a squat. **That measurement is impossible on
the synthetic clip**, because Finding 1 shows the detected signal is
uncorrelated with the truth. There is no lag to measure when there is no
tracking.

**So the filter cannot be retuned honestly yet.** It needs real footage.

What can be said: a 16% jitter reduction is a weak justification for a filter
that adds any lag at all, and the user has reported lag. The recommendation is
to make it **configurable and default it off**, so it can be measured properly
when the camera works — rather than removing it outright on the strength of a
measurement that could not be finished.

---

## Finding 3 — what is safe to build now, and what is not

The distinction that matters: some work depends on tracking quality, and some
does not.

### Safe to build without a working camera

These are **pure functions of landmark input**. They can be unit-tested with
synthetic landmark sequences — arrays of coordinates, not rendered images — so
the rendering problem in Finding 1 does not touch them:

- Hysteresis state machine with sequence validation
- Repetition counting
- Dempster centre-of-mass computation
- Jump detection with free-fall validation
- Median filter for spike and limb-swap rejection
- Viewpoint gate
- Subject-switch detection
- 100 ms gap reset

Feeding these hand-written landmark arrays tests the *logic* exactly, and the
logic is what Round 3 settled with published numbers.

### Not safe to decide without a working camera

- Filter constants
- Whether our extra smoothing should exist at all
- Absolute angle thresholds for a specific user
- Any accuracy claim
- RTMPose against MediaPipe

---

## Method note: why synthetic landmarks are legitimate where synthetic video is not

The distinction is worth stating because it is the difference between valid and
invalid testing here.

**Synthetic video** asks a trained model to interpret an image unlike anything
in its training distribution. The model's response tells you about the gap
between the image and the training data — not about the model, and not about
anything downstream.

**Synthetic landmarks** skip the model entirely. A hysteresis state machine
given the coordinate sequence of a squat must count one repetition. That is a
property of the state machine, and it holds regardless of where the coordinates
came from.

The same reasoning appears in `AGENTS.md`: the Dart core is pure and testable on
a laptop precisely because it takes landmarks, not pixels.
