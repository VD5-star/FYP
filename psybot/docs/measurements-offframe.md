# Off-frame landmarks, the seen-before rule, and joint anchoring

The reported symptom: "when I jump and leave the edges, the skeleton still
shows random, illogical marks."

## The cause

MediaPipe places landmarks that have left the picture at coordinates **outside
the image**, and still reports high confidence for them. Measured on a real
photograph shifted until the left side crossed the edge of a 1280x720 frame:

| landmark | x | y | past the edge | visibility |
|---|---|---|---|---|
| leftElbow | 1286 | 363 | 6 px | 0.98 |
| leftWrist | 1341 | 363 | 61 px | 0.94 |
| leftPinky | 1349 | 362 | 69 px | 0.89 |
| leftAnkle | 1333 | 537 | 53 px | 0.75 |
| leftFootIndex | 1355 | 546 | 75 px | 0.54 |

Nine landmarks outside the image, the worst 75 px past the edge. The drawing
gate was 0.2, so every one of them was drawn.

## `presence` does not solve this, and it looked like it would

MediaPipe's landmark proto documents two separate scores:

- `visibility` - whether the landmark is **occluded** by other objects
- `presence` - whether the landmark is **present on the scene**

The second reads exactly like an out-of-frame signal, and the Python Tasks API
does expose it. Measured on landmarks genuinely outside the image:

    presence 0.73 - 0.99

It does not detect off-frame at all. **No model output distinguishes "outside
the picture" from "hidden behind something."** The only reliable test is
geometric: is the coordinate inside the image?

This is worth recording because the field name strongly implies otherwise, and
a future reader would reasonably try it again.

## The rules

Applied in order, per landmark, per frame:

| condition | action |
|---|---|
| coordinate outside the image | **discard**, and erase the landmark's history |
| hidden, observed 20+ times before | **hold** at the last trusted position, decaying |
| hidden, not observed 20 times | **discard**, invent nothing |

### Why leaving the frame erases the memory

A limb that walks out of shot is not occluded, it is gone, and where it went is
unknowable. Holding its last position would pin a hand to the edge of the
picture. When the subject returns, the held position would be stale by however
long they were away. So crossing the boundary clears that landmark's history,
and it must earn its 20 observations again.

### Why 20 frames and not one

The model reports high confidence for limbs it has never seen - measured at
0.97 for a wrist a full torso length out of place. A single confident frame
therefore cannot establish that a limb exists. Twenty frames is about two
thirds of a second of genuine, in-frame, skeleton-consistent observation.

Doubtful frames do not count towards the total, or a limb could be established
from the very frames in which it was missing.

## The edge margin was wrong first time

A margin is needed so a landmark sitting on the boundary does not flicker as
noise pushes it across the line. The first value chosen was 2%, by reasoning
rather than measurement. On a 1280-wide frame that is 26 px - **twenty-four
times the measured 1.09 px noise floor** - and it let two genuinely off-frame
landmarks through:

| margin | pixels | off-frame landmarks still admitted |
|---|---|---|
| 0.0% | 0.0 | 0 |
| **0.2%** | **2.6** | **0** |
| 0.5% | 6.4 | 1 |
| 2.0% | 25.6 | 2 |

Settled at 0.2%.

## Drawing

Bones are clipped to the image with Liang-Barsky, so a bone with one end
outside is drawn as far as the edge and no further. A bone drawn to an
unclipped endpoint skews towards wherever the invented coordinate landed, which
is what made the skeleton sit beside the body rather than along it.

Three drawing steps were found, by measurement, to ignore the discard flag and
draw anyway:

| step | pixels drawn with everything discarded |
|---|---|
| hand and foot boxes | 1778 |
| angle rings and labels | 2256 |
| corner brackets | 33 |
| **after the fix** | **0** |

The trunk was already correct. The angle labels mattered most: printing a
number next to a joint asserts that a measurement was made.

## Joint anchoring

Once a joint's position is known it should stay there rather than being nudged
somewhere slightly different every frame.

### The margin is thin, so this needed measuring

Frame-to-frame movement, in torso units:

| joint | still, p95 | moving, median | ratio |
|---|---|---|---|
| leftKnee | 0.0074 | 0.0119 | **1.6x** |
| rightKnee | 0.0037 | 0.0092 | 2.5x |
| leftElbow | 0.0070 | 0.0225 | 3.2x |
| leftWrist | 0.0072 | 0.0529 | 7.4x |

The left knee has only a 1.6x margin, and the knee is what every threshold in
this engine reads. A plain deadband sized to remove the shiver would eat part
of a slow squat.

### The design measurement chose

Hold the joint while quiet, release it completely once movement clears the
noise floor, blend between the two so there is no visible snap.

| design | still jitter | cut | knee error, squat |
|---|---|---|---|
| none | 0.00191 | - | - |
| deadband 0.0050 | 0.00095 | 50% | 0.047 deg |
| deadband 0.0074 | 0.00061 | 68% | 0.080 deg |
| adaptive 0.0030/0.010 | 0.00058 | 69% | 0.066 deg |
| **adaptive 0.0050/0.015** | **0.00032** | **83%** | **0.146 deg** |
| adaptive 0.0074/0.020 | 0.00018 | 91% | 0.285 deg |

`0.0050/0.015` removes 83% of the shiver for 0.146 degrees of knee error,
against a knee-angle noise floor of 0.66 degrees - about a fifth of the noise
it competes with. The stronger setting removes 91% but triples the error and
reaches 1.4 degrees on the fast clip.

### A running-mean leash was tried and rejected

The blend has no memory of a centre, so in principle it could random-walk under
noise. A leash was added to hold the joint at the running mean of what had been
observed. Measured drift over one second:

| design | drift |
|---|---|
| blend only | 4.37 px |
| leash only | 4.37 px |
| blend towards mean | 4.37 px |

**Identical.** The drift is not a random walk: at the real noise floor, 1.08% of
frames exceed the release threshold by pure chance - about seven spikes in six
hundred frames - and each frees the joint. No leash can prevent that, because
the release threshold is the only thing separating noise from real movement.

The leash also cost 0.75 px of lag and produced a non-monotonic release curve
([0.5, 0.25, 0.75, 1.0]) when combined with the blend: a joint moving slightly
followed *less* than one barely moving. Two mechanisms setting the same value
fought each other. Removed.

## Counting stops rather than fabricating

When the joints an exercise depends on are all discarded, the angle is passed
to the counter as `None` - meaning "no information", which it already handles -
and the display reads "step back - I cannot see your knee". Completed reps are
kept. A number computed through a discarded landmark would mean "here is a
measurement", which would be false.

## A recorded finding that did not reproduce

`pose_sim.py` documented that `lite` was both faster **and more accurate** than
`full` (knee sigma 0.43 vs 0.79 degrees), and that was the stated reason `lite`
is the default.

Re-measured across three real clips and four joints - twelve comparisons:

    wins:  heavy 9,  lite 2,  full 1

On the still clip's left knee, `full` measured 0.207 deg against `lite` at
0.711. The original finding appears to have rested on a single joint on a
single clip.

`lite` remains the default, but on **speed**, measured end to end through the
whole pipeline at 1280x720:

| model | per frame | fps |
|---|---|---|
| lite | 39.5 ms | 25.3 |
| full | 51.0 ms | 19.6 |
| heavy | 150.1 ms | 6.7 |

`full` costs 26% of the frame rate for roughly half a degree. Whether that is
worth it depends on the machine, so it is now `--model` rather than a fixed
choice.

## A throughput scare that was a measurement error

The synthetic main-loop harness reported 11 fps after these changes, against
31.2 fps recorded earlier - an apparent severe regression.

Profiled, the new code costs under 1 ms per frame in total: occlusion 0.406 ms,
apply_model 0.192 ms, anchor 0.123 ms, clipping 0.012 ms per bone.

Stashing every change and re-running the same harness on the committed baseline
gave **10.0 fps** - slower than the new code. The 31.2 figure had been taken
under different machine load and was never a valid comparison. Steady-state
per-frame timing is the reliable measure, and it gives 25.3 fps.

## Not measured

- All of this uses a real photograph transformed synthetically. **No real
  person has jumped in front of the camera and been checked**, which is the
  test that actually matters for the reported symptom.
- The 20-frame threshold is reasoned from the model's confidence behaviour, not
  fitted to a distribution of real appearances and disappearances.
- The occlusion tolerances (0.35, 45/s, 1.5 s) remain inferred, not measured
  against real occlusion cases.
