# Stability, occlusion and finger reading — measured

Three reported problems, each traced to a measured cause before anything was
written.

## 1. The figure shivers and does not hold together

### Cause: nothing enforces a skeleton. At all.

Verified at source level rather than from documentation. MediaPipe's
`LandmarksSmoothingCalculator` proto offers exactly three options:

```
oneof filter_options {
  NoFilter        no_filter       = 1;
  VelocityFilter  velocity_filter = 2;
  OneEuroFilter   one_euro_filter = 3;
}
```

All three operate on **each landmark independently**. There is no bone-length
term, no kinematic chain, no coupling between joints of any kind. ML Kit exposes
no smoothing control whatsoever.

`pose_world_landmarks` does not help. Despite the name it is not a world
*model* — it is an alternate output space (metric 3D, origin between the hips)
regressed per frame, with no temporal or skeletal consistency guarantee.

### The size of the problem, on our own footage

| Clip | bone variation | worst bone, peak to peak |
|---|---|---|
| `real_still.mp4` (subject not moving) | 0.72% | **6.9%** |
| `real_curl.mp4` (arm movement) | 12.59% | **179%** (forearm) |

A forearm that changes length by 179% is not a forearm. Filtering each point
independently smooths the *picture* while the *body* still comes apart, which is
exactly the reported symptom.

### The fix, and why it is free

Learn the body once, then enforce it — the order the user asked for:

1. **Calibrate** over ~2 seconds: median bone length, normalised by torso
   length, averaged left against right.
2. **Enforce** on every later frame: keep each bone's *direction* from the
   detection, substitute the *calibrated length*.

Lengths are stored relative to torso, never in pixels — a pixel length is only
correct at the distance it was measured at.

This is the adjustment mechanism from BLAPose (arXiv:2410.20731), which
preserves bone orientations while substituting lengths. The research
contribution there is a learned length predictor; the adjustment is arithmetic.

**Joint angles are functions of bone directions alone**, so this cannot change
any angle. Measured across every joint and every clip:

```
angle change: 0.000 degrees
```

### The defect that measurement found

The first implementation measured each bone's direction from its
**already-moved** parent, mixing two coordinate frames and compounding error
down the chain:

| Joint | angle change, cumulative (wrong) | consistent frame (right) |
|---|---|---|
| leftElbow | mean 9.84°, **max 175.29°** | 0.000° |
| rightElbow | mean 14.78°, **max 172.25°** | 0.000° |
| leftKnee | mean 0.50°, max 4.72° | 0.000° |

The error is worst at the end of the chain — the wrist — which is precisely what
the elbow angle depends on. A non-zero number in the test now means this defect
has returned.

### Measured effect

| Clip | bone variation raw | constrained | jitter raw | constrained |
|---|---|---|---|---|
| still | 0.72% | **0.20%** | 0.353 px | **0.246 px** (−30%) |
| curl | 12.59% | **3.45%** | | |
| jump | 1.08% | **0.42%** | | |
| sway | 0.71% | **0.23%** | | |

### One clip that looks like a failure, and is not

`real_squat.mp4` went 6.76% → 12.90%. Investigated rather than tuned away.

After the constraint, **bone variation equals torso variation exactly** on every
clip — each bone is now a fixed multiple of torso length, so the scale carries
all residual. On that clip the measured torso length swings between 104 and 150
pixels, a 37% range, smoothly rather than noisily. It is a still photograph
under rigid transforms and the transform *scales the figure*, so there is no
stable scale to constrain against.

**Smoothing the torso was tried and rejected on measurement.** On a real subject
the scale is already stable at 0.20%, and every filter tested made it worse:

| Method | torso sd on `real_still.mp4` | max deviation |
|---|---|---|
| raw | **0.20%** | — |
| running median 5/9/15 | 0.21–0.22% | 0.59% |
| one-euro (3 settings) | 0.25–0.29% | up to 1.22% |

Filtering an already-clean signal only adds lag and error. The scale is used
raw, and the honest limit is that this assumes the subject's distance changes
slowly — true of a person exercising, false of a scaled photograph.

### Why not full IK

The user asked for IK. What shipped achieves the same bone-length guarantee in
closed form, in one pass, with no iteration and no solver — and is provably
angle-preserving, which an iterative fit is not. The remaining benefit of true
IK is enforcing *joint limits*, which is a separate concern from the reported
problem, and the user also asked that this stay simple enough to port to
Flutter. It is pure arithmetic on landmark arrays and ported to Dart unchanged.

## 2. A hidden limb misbehaves instead of predicting sensibly

### Cause: the model's confidence is sometimes a lie

Both SDKs always emit all 33 landmarks and are trained to regress plausible
positions for hidden ones. Measured by covering part of a real photograph:

| Occluded region | landmark | position error | **model confidence** | our 0.5 gate |
|---|---|---|---|---|
| lower legs | leftKnee | 78 px (0.53 torso) | 0.12 | rejected ✓ |
| lower legs | leftAnkle | 174 px (1.18 torso) | 0.06 | rejected ✓ |
| **left arm** | leftElbow | 43 px (0.29 torso) | **0.99** | **accepted ✗** |
| **left arm** | leftWrist | **149 px (1.01 torso)** | **0.97** | **accepted ✗** |

The wrist moved a **full torso length** and still reported 0.97. The gate did
not reject it, so an invented position was consumed as a measurement.

The legs behaved correctly, which is worse than never being right — the
confidence cannot be relied on in either direction.

### The fix: a second opinion that does not trust the first

Two independent tests, neither using the model's confidence:

- **Bone length.** The calibrated model says how long a forearm is. Disagreement
  beyond 35% means the position is wrong regardless of confidence. (35% because
  foreshortening genuinely shortens a limb pointing at the camera; the measured
  failure was a full torso length, far outside it.)
- **Teleport.** Movement beyond 45 torso-lengths per second is not tracking. For
  scale, an explosive jump peaks at 17.5 /s at the hips.

A failing landmark is **held** at its last trusted position — not extrapolated,
which diverges without limit and looks confident while being wrong — and its
confidence decays. Verified:

```
error at the moment of occlusion:  149.9 px  ->  0.0 px  (held)

seconds held   confidence   usable
       0.2         0.84      yes
       0.6         0.57      yes
       0.8         0.44      no
       1.6         0.00      abandoned
```

Held limbs are drawn **dashed** and their joints hollow, so an inferred position
never looks like a measured one. Downstream measurements **abstain** rather than
report a fabricated number.

The distinction observed: interpolating a briefly hidden joint between two
confident observations is inpainting; asserting a position for a joint hidden
for seconds is hallucination. This holds, decays, then abstains.

## 3. Finger reading and the stop gesture

### The blocker: ML Kit has no hand API

Confirmed from the Google Maven group index — there is no `hand-*` artifact in
any version. ML Kit Pose gives 3 points per hand (wrist, thumb, index, pinky)
with no finger joints, from which finger state cannot be derived.

`com.google.mediapipe:tasks-vision` reached **1.0.0 stable in July 2026** and
gives 21 landmarks per hand, plus separate `visibility` and `presence` signals
that ML Kit lacks entirely — directly relevant to problem 2 above. **The user
chose to migrate the Android layer to MediaPipe.** That migration is *not* done
here and is recorded as outstanding, because it cannot be tested without a
phone.

### The crop is mandatory, not an optimisation

Measured by shrinking a real hand photograph into a 1280×720 frame:

| Hand width | full frame | wrist crop |
|---|---|---|
| 300 px | yes | yes |
| 200 px | **NO** | yes |
| 140 px | **NO** | yes |
| 70 px | **NO** | yes |
| 40 px | **NO** | yes |
| 30 px | NO | NO |

A person framed head to toe has a hand about **40–70 px** across — where
full-frame detection **fails completely**. The pose model's wrist landmark is
used to cut and upscale a box. Without this the feature does not work at
exercise distance at all.

Cropping does **not** make inference cheaper (18.5–19.3 ms at every crop size,
because MediaPipe rescales to its own fixed input). It buys accuracy, not speed.

### Cost, and why it samples intermittently

Hand inference costs 11.2 ms/frame, which would take the simulator from 36.8 to
**26.0 fps** — a 29% loss for a feature idle almost all the time.

The gesture must be held 1.5 s, so every frame is pointless:

| Sampling | amortised cost | samples per 1.5 s hold |
|---|---|---|
| every frame | 11.2 ms | 52 |
| **every 4th** | **1.5 ms** | **13** |

Measured end to end with everything enabled: **36.5 fps**.

### The gesture

Numbering as specified: pinky=1, ring=2, middle=3, index=4, thumb=5. Stop is
**1 and 5 extended, 2/3/4 folded** — a shaka — held 1.5 s, then the program
quits.

Finger state is tested by **distance from the wrist**, comparing the tip against
the finger's own middle joint. Rotation-invariant, unlike a y-coordinate test,
which fails the moment the hand tilts. Verified on real photographs: a victory
sign reads `[3, 4]`, a thumbs-down reads `[5]`.

The 1.5 s hold is not decoration — arm movement is part of several exercises, so
firing on one frame would close the program mid-set. A progress ring is drawn so
the command can be seen and abandoned.

## Not established

- The bone model assumes the subject's distance from the camera changes slowly.
  Measured true for a person exercising; false for a scaled photograph.
- Occlusion thresholds (35% bone tolerance, 45 /s teleport, 1.5 s hold) are
  **inferred, not measured** against a distribution of real occlusions. The
  149 px failure they were sized against is a single measurement.
- The hand distance test used one photograph shrunk synthetically, not a real
  person standing at real distances.
- **Rep counting on a real moving human remains unvalidated**, as before.
- The Android migration to MediaPipe is decided but **not done** — no phone.
- `PoseSmoother` lag is still unmeasured: it was attempted against the live
  camera but needs a person in front of it.
