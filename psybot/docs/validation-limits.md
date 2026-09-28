# What cannot be validated without a camera, and why

Three separate attempts were made to validate repetition counting on real
photographic imagery. **All three failed**, for the same underlying reason, and
recording why is more useful than recording a number that would have been
meaningless.

## The attempts

### 1. Vertical compression of the whole figure

Squashing the image vertically was meant to simulate a squat.

**Result:** knee angle varied over **7 degrees** across the entire clip. Both
counters correctly reported zero.

**Why it failed:** compressing a picture of a person does not bend their knee.
It shortens every segment uniformly, which changes no joint angle. The knee
angle is determined by the *directions* of the thigh and shin, and a uniform
scale preserves direction.

### 2. Anatomical warp of the lower body

Abandoned before measurement, on inspection of where the subject's joints
actually are. The person in MediaPipe's test photograph is **seated on the
ground** — measured landmark positions put the right knee at (357, 490) and the
right ankle at (350, 608), with the hips at the same height as the knees.

**No warp turns a seated person into a squatting one.** There is no standing
posture to return to.

### 3. Rotating the forearm about the elbow

The most promising attempt: the subject's left elbow is extended at 172 degrees
with confidence 1.00, so rotating the image region beyond the elbow should
produce a genuine flexion on real photographic texture.

**Result, and the reason this document exists:**

| | |
|---|---|
| Correlation, detected elbow angle vs true rotation | **0.004** |
| Detected range | 30–180° |
| True range | 100–180° |
| Frames flagged untrustworthy by the reliability gates | **111 of 150** |

The rotation tore the body apart rather than bending an arm. A rigid rotation of
a circular image region cannot respect anatomy: it drags background, shoulder
and torso pixels along with the forearm, and it cannot synthesise the occlusion,
foreshortening and shading change that a real bending arm produces.

**Both counters were therefore measuring nothing.** The velocity counter
reported 5 of 5, which looked like a success and was not — it was a plausible
number produced from corrupted input, which is exactly the kind of result that
is worse than an obvious failure.

## The rule this establishes

**A still photograph can validate tracking, but not movement.**

Translating or scaling a photograph tests whether the model follows real
texture, and it does: correlation 0.997 for a translated figure
(`docs/validation-real-imagery.md`). But every *joint angle* stays fixed under
those transforms, because angles depend on the relative directions of body
segments and rigid transforms preserve those.

Changing a joint angle requires changing the *content* of the image — new
occlusions, new shading, new silhouette — and that cannot be faked from one
frame. It requires either video of a real person or a rendered 3D human, and
rendered humans are what Round 4 showed MediaPipe does not track.

## The genuinely useful result

**The reliability gates worked exactly as designed.**

They flagged 111 of 150 frames as untrustworthy, citing impossible limb-length
changes and impossible landmark speeds. That is a true report: the input *was*
corrupt.

This is the first end-to-end evidence that the fast gates catch bad tracking on
real image data rather than only on hand-constructed test cases — and they
caught it in one frame, where the visibility score they supplement would have
taken 233 ms to react.

Had this system been shipped without those gates, it would have counted five
confident repetitions from a body being torn apart.

## What remains unvalidated

**Repetition counting on a real moving human.** Both counters. Every synthetic
validation of them uses hand-constructed landmark sequences, which test the
logic exactly but assume the landmarks are correct.

This is the single measurement the new camera settles, and it needs nothing
elaborate: perform five squats in front of `pose_sim.py --exercise squat` and
compare the count to reality.
