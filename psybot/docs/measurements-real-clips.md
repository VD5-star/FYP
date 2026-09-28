# Survey of 28 real clips

Everything before this was measured on a single photograph, transformed. The
project rule says a still image proves tracking, not movement, so the numbers
could not settle how the engine behaves on real people.

28 free-licence clips were downloaded and run through the real pipeline:
people walking toward and away from the camera, putting a hand behind the back,
turning, stretching, dancing, squatting, lifting.

## The reported bug, confirmed

The report was that arm length is measured wrongly when the subject changes
position, even at the same apparent size.

The same person, same scale, pasted at seven places in one frame:

| position | leftElbow | leftWrist | leftKnee |
|---|---|---|---|
| centre | 0.5705 | 0.4803 | 0.7342 |
| left | 0.5899 | 0.4870 | 0.7466 |
| high | 0.6158 | 0.4883 | 0.7671 |
| low | 0.5715 | 0.4598 | 0.6654 |

Worst spread from position alone: **14.2%**, against 3.45% honest movement
variation. The body never changed.

The cause is perspective: a person at the edge of the frame is viewed
obliquely, so their limbs really are foreshortened in the picture.

## Four defects found, all by measurement

### 1. Calibration gates were far too strict

The torso gate rejected a whole frame if any one of the four torso landmarks
fell below 0.5 confidence:

| clip | frames rejected |
|---|---|
| dance_1 | 196 of 200 (98%) |
| stretching_2 | 195 of 200 (98%) |
| turning_1 | 135 of 200 (68%) |
| walk_toward_3 | 0 of 200 |

The body model never built on those clips, so bone lengths ran completely
unconstrained at 40-55%. Confidence is not a statement about position - the
same lesson as the off-frame work. Now 0.2 plus a geometric in-frame check.

The per-bone gate was 0.5 for the same bad reason. Measured across eight
clips, **lower is better on both counts**, because the statistic is a median
and a median wants more samples, not purer ones:

| gate | clips where the model built | bone spread |
|---|---|---|
| **0.2** | **6 of 8** | **6.3%** |
| 0.3 | 6 of 8 | 10.3% |
| 0.5 | 5 of 8 | 11.0% |
| 0.6 | 5 of 8 | 20.1% |

### 2. A held limb was stored in pixels

On a clip where the torso spanned 33 to 182 px, a held elbow gave a bone-ratio
spread of **553.9%**, against **29.2%** over the frames where the same bone was
actually seen. The limb had not moved; the reference had doubled.

Holds now store an offset from mid-hip in torso units, rebuilt each frame.
That introduced 16 off-frame escapes of its own - scaling an offset can throw a
limb past an edge after the out-of-frame rule has already run - so the rebased
position is re-checked.

### 3. The anchor could push a landmark out of frame

10 landmarks across 28 clips ended up 0.9 to 2.2 px past an edge, having
already been passed as drawable. The anchor runs after the check, so it can
undo it. Held positions are now clamped inside the frame when the detection was
inside, and deliberately **not** when the detection was genuinely outside,
which would disguise it as visible.

### 4. No guard looked at joint angles

199 cases of a tracked angle changing more than 40 degrees between consecutive
frames, worst 153.8 degrees. Both existing guards missed every one, and the
reason is structural: they test each landmark in isolation. A knee swinging 153
degrees moves perhaps half a torso, far under the 1.8 torso-per-frame teleport
limit, and its bone length can stay plausible throughout.

4337 consecutive-frame angle changes across ten clips:

| percentile | deg/frame |
|---|---|
| p50 | 0.7 |
| p90 | 7.0 |
| p99 | 51.5 |
| p99.9 | 128.7 |
| max | 163.3 |

A sevenfold gap between p90 and p99 is two populations, not a continuum.
`MAX_ANGLE_RATE = 500 deg/s` is 20 deg/frame at 25 fps.

The guard **rejects rather than corrects**: a rejected angle becomes "no
information", which every consumer already handles. A rejected value is not
recorded as the reference, so one bad frame cannot drag the comparison.

## Results

| metric | before | after |
|---|---|---|
| bone spread, median | 52.8% | 47.4% |
| off-frame drawn | 10 | **0** |
| angle jumps over 40 deg | 199 | **9** |
| woman-doing-yoga | 17.8% | **2.4%** |
| exercise_3 | 29.5% | **4.6%** |
| walk_toward_3 | 3.7% | **2.4%** |
| squat_1 | 21.4% | **11.1%** |
| hands_behind_2 | 82.1% | **52.6%** |
| turning_1 | 322.0% | **160.6%** |

## Hidden-arm behaviour, the scenario asked for

Traced frame by frame on the hand-behind-back clips. The state sequence is
`seen -> held -> discarded -> seen`, and **`discarded -> held` never occurs** on
either clip: a limb is never revived from nothing.

## Tried and rejected, with the measurement

**3D world landmarks** for the position bug. They are *worse*: 27.3% spread
against 5.4% for pixels on distance change, and no better on position. The
name suggests they should be position-independent; measured, they are not.

**A shoulder-width fallback** for waist-up framing, where the hips are outside
the picture and torso length cannot be measured at all. It made turning_1 four
times worse (44.1% to 164.9%) by mixing two scales within one clip.

**Faster calibration refresh.** 90 frames gives 19.5% against 3.5% at 300 - a
refresh throws away a settled median and rebuilds it from whatever the subject
happens to be doing, so frequent refreshes keep the model permanently
half-learned.

**Anchor thresholds and the 20-frame trust threshold** were swept across five
settings each and make no measurable difference on real footage. Left alone.

## Adaptation speeds, tuned

| setting | was | now | evidence |
|---|---|---|---|
| calibration learn window | 60 | **30** | 2.4% vs 3.5% bone spread, ready twice as fast |
| calibration refresh | 300 | 300 | 90 frames measured much worse (19.5%) |
| anchor quiet/release | 0.005/0.015 | unchanged | no measurable difference across 6 settings |
| trust threshold | 20 | unchanged | no measurable difference across 5 settings |

A longer learn window is not a better median: it spans more of the subject's
movement, so more foreshortened samples drag it.

## Limitations, stated not hidden

- **Waist-up clips cannot be constrained.** On `stretching_2` the hips sit 34%
  below the bottom edge in every frame. There is no torso to measure, no scale,
  and no body model. Not a bug - a framing requirement.
- **The worst clip (497%) is out of range.** It spends 96% of its frames with
  the body under 400 px, below the measured detection floor, and as small as
  74 px. The landmark noise there is the model's.
- **Both remaining high-spread clips are one of those two cases.**
- The clips are third-party footage at unknown camera settings. Motion blur,
  compression and frame rate all vary and none were controlled.
- No measurement here used the user's own camera or a person the engine will
  actually be used on.
