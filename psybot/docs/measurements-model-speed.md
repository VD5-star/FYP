# Model speed and accuracy, measured

Research into speed optimisation identified model and input resolution as the
steepest available lever — steeper than any filtering change. This measures it
on our own imagery rather than trusting published tables.

## Method

MediaPipe's official test photograph of a real person, 1000×667, with sensor
noise (σ = 2.5 grey levels) added per frame. 80 detections per model in IMAGE
mode, so each frame is detected independently with no temporal tracking to
flatter the result.

Laptop CPU. **These absolute numbers will not transfer to a phone**; the
*ratios* are the useful part.

## Result

| Model | Median | p95 | Jitter | Knee angle σ | vs heavy |
|---|---|---|---|---|---|
| **lite** | **25.2 ms** | 26.8 ms | 0.70 px | 0.43° | 4.0 px mean, **0.2° on knee** |
| full | 30.6 ms | 33.5 ms | 0.72 px | 0.79° | 3.5 px mean, 1.3° on knee |
| heavy | 70.7 ms | 77.3 ms | 0.46 px | 0.22° | — |

## What this changes

**`lite` is 18% faster than `full` and no less accurate for our purpose.**

That is not the expected result, and it is worth being precise about what it
does and does not say:

- On **jitter**, `lite` and `full` are indistinguishable (0.70 vs 0.72 px). The
  difference is well inside the run-to-run variation.
- On **knee-angle stability**, `lite` is *better* (0.43° vs 0.79°).
- On **agreement with `heavy`**, `full` is marginally closer in mean landmark
  position (3.5 vs 4.0 px) but `lite` is much closer on the knee angle that
  actually drives every threshold (0.2° vs 1.3°).

So the ordering flips depending on what you measure, which means **`full` has
no measurable advantage for this application.** It costs 5.4 ms per frame for
nothing.

`heavy` is genuinely more stable — jitter 0.46 px, knee σ 0.22° — but at
**2.8× the latency of `lite`**. Against a measured noise budget with 9× margin
(`docs/measurements-real-human.md`), buying more stability we do not need at
that price is the wrong trade for a project whose stated priority is
responsiveness.

## Recommendation

**Use `lite`.** Measured faster and equally accurate where it matters.

Two caveats stated plainly:

1. **One photograph, one pose, one subject.** Model differences are most likely
   to appear in hard poses — heavy occlusion, unusual angles, partial framing —
   and a single upright figure does not sample those. The honest claim is "no
   measurable advantage *on this image*", not "never any advantage".
2. **Laptop CPU, not a phone.** The ratio should roughly hold, since it comes
   from model size, but Android's GPU delegate and NNAPI can change relative
   costs in ways a CPU benchmark cannot predict.

## Why no other model was adopted

RTMPose-t reports 9.02 ms on a Snapdragon 865 (ncnn FP16) against RTMPose-s at
13.89 ms, and is genuinely deployable on Android under Apache-2.0. It is the
only credible alternative found.

It was not adopted, for one decisive reason: **it produces 17 COCO keypoints
where MediaPipe produces 33.** We would lose the foot, hand and facial
landmarks that framing assessment, the Dempster centre-of-mass model and the
extremity boxes all depend on — to fix a bottleneck we have not measured on a
phone.

Others considered and rejected:

- **MoveNet Lightning** — faster, but also 17 keypoints.
- **YOLO-pose** — 640×640 input, roughly 20× RTMPose-t's FLOPs, and AGPL-3.0,
  which is a licensing problem for a closed-source app.

**Revisit only after measuring that pose inference is actually the bottleneck on
a real device.** On this laptop, `lite` at 25 ms already clears 30 fps.
