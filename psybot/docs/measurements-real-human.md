# The noise floor, measured on a real human

This is the number every hysteresis threshold in the system rests on, and until
now it was unmeasured. `docs/research-round4-notes.md` records why: the
synthetic clips could not provide it, because MediaPipe never tracked the
rendered figure at all (correlation −0.04 with the driving signal).

## Method

Source image: MediaPipe's own official test asset, `pose.jpg` — a photograph of
a real person, 1000×667.

A still camera pointed at a still person does **not** produce a static image, so
copying one frame 150 times would measure nothing. Two real effects were
reproduced:

- **sensor noise** — Gaussian, σ = 2.5 grey levels
- **sub-pixel camera shake** — σ = 0.4 px translation per frame

Both are conservative for a phone propped on a table.

## First, confirming the model works on a real person

| | |
|---|---|
| Landmarks above 0.5 confidence | **33 / 33** |
| Mean visibility | **0.992** |

Compare with the rendered figure, which produced high torso confidence while
the positions wandered independently of the geometry. **The problem was the
synthetic imagery, not the model.**

## The measurement

Two modes were run, and the difference between them matters.

| | VIDEO mode | IMAGE mode |
|---|---|---|
| Per-landmark jitter | 0.30 px | **1.09 px** |
| Knee angle σ | 0.11° | **0.66°** |
| Knee angle range | 0.5° | 3.3° |

VIDEO mode applies MediaPipe's internal temporal tracking and One-Euro filter.
IMAGE mode detects each frame independently, with no memory.

**The honest number is IMAGE mode: 1.09 px, 0.66°.** The VIDEO figure is 3.6×
better, but that improvement comes from smoothing, and quoting it as the noise
floor would be quoting the filter's output as if it were the sensor's input.

Torso length was 147 px, so 1.09 px is **0.74% of torso length** — the
scale-free form, and the one that transfers to other distances and resolutions.

## What this means for the thresholds

The stress tests in `test_movement.py` measured the counter's tolerance:
reliable to about **10 px** of per-landmark noise, breaking down at 15 px.

| | |
|---|---|
| Measured noise | 1.09 px |
| Budget | 10 px |
| **Margin** | **9×** |

And against the published literature — the RTMPose-versus-Qualisys validation
reporting knee-angle limits of agreement of −6.7° to +11.9°:

| | |
|---|---|
| Measured angle σ | 0.66° |
| Published LoA | ±12° |

The 60° dead zone is therefore **conservative by a wide margin**, not
marginally safe. That was the right call regardless, because the published
figure describes a *moving* subject with real occlusion and clothing, while
this measures a still one.

## Limits of this measurement — stated plainly

1. **One subject, one pose, one photograph.** It does not sample body shapes,
   clothing, lighting or camera quality.
2. **A still person is the easy case.** Motion blur, self-occlusion and
   foreshortening all degrade tracking, and none are present here.
3. **Synthetic noise, not a real sensor.** The model is reasonable but it is a
   model. A real camera adds rolling shutter, compression artefacts,
   auto-exposure hunting and auto-focus breathing.
4. **Laptop CPU, not a phone.**

So the honest claim is: **under ideal conditions the noise floor is roughly 9×
inside budget.** Real conditions will be worse — but 9× is a large amount of
headroom to spend, and it means the design is not operating near its limit.

The number to replace this with, when the camera arrives: the same measurement
on a real person genuinely holding still, and then again mid-movement.
