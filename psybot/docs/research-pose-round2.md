# Round 2 — measured on this machine

Round 1 collected other people's numbers from other people's hardware. This
round measures on this laptop. **No production code was changed** — only a
benchmark harness and a test-clip generator were added.

---

## First, an honest account of the method

**The laptop camera became unavailable mid-round.** It is present and enabled in
Device Manager and the privacy setting permits access, but every OpenCV backend
reports "camera index out of range". Resetting the device needs administrator
rights, which are not available here.

Rather than stop, the measurement moved to **generated test clips**. That turns
out to be methodologically better for the central question, and worse for
another — both stated plainly below.

### Why a synthetic clip is better for measuring the noise floor

The decisive question is: *how much does a joint angle wander when the body is
not moving?* On a still subject the true variance is zero, so **everything
observed is model error** — no reference system is needed.

Filming a real person "holding still" cannot deliver that cleanly, because a
person breathes and sways. Their genuine micro-movement is inseparable from the
model's error.

The clips render the figure from **identical geometry on every frame**, with
realistic webcam noise added on top (Gaussian, σ = 3.5 grey levels — a modelling
choice, not a measurement). True variance is exactly zero by construction.

Three clips, 300 frames each at 30 fps: `still`, `squat` (a known knee-angle
cycle) and `arms`.

A second and equally important benefit: **every model sees identical input.**
Two camera runs never see the same frames, so part of any difference between
them is the subject having moved. Replaying one clip removes that variable.

### Where this method fails, and it matters

**The figure is rendered, so it is out of distribution for models trained on
photographs.** This is not a small caveat — it invalidated one of the three
measurements outright, as shown below. Every number here needs confirmation on
real camera footage before it is trusted.

---

## Measurement 1 — joint-angle noise floor, still subject

The number that decides whether hysteresis thresholds are viable.

| Model | fps | inference (median) | detection | **angle std, median** | worst joint |
|---|---|---|---|---|---|
| lite | **53.7** | **11.8 ms** | 100% | 1.52° | 3.07° (right knee) |
| **full** | 42.5 | 17.0 ms | 100% | **0.51°** | 0.97° (left elbow) |
| heavy | 14.8 | 58.1 ms | 86.2% | 8.30° | 42.71° (left hip) |

**`full` is the most stable by a factor of three over `lite`, and sixteen over
`heavy`.**

### The `heavy` result is an artefact, not a finding

A model that is documented as *more* accurate measuring sixteen times worse is
not credible, so it was investigated rather than reported.

First hypothesis: `heavy` lost tracking on 14% of frames, and each re-acquisition
produces a jump that a standard deviation counts as jitter. Testing this by
measuring only within continuous tracking segments:

| Model | continuous frames | elbow std *within* a segment |
|---|---|---|
| lite | 290 | 1.50° |
| full | 290 | **0.97°** |
| heavy | 221 | 6.61° |

So the jumps explain part of it — 42.71° fell to 6.61° — but `heavy` is still
worse. The real cause was found by inspecting its confidence outputs:

| Landmark | `heavy` mean visibility |
|---|---|
| left shoulder | 1.000 |
| left hip | 0.997 |
| nose | 0.994 |
| **left knee** | **0.291** |
| **left ankle** | **0.272** |

`heavy` is *correctly* reporting that it does not believe the legs of a rendered
cartoon figure. It is the most discriminating of the three models, and the
synthetic clip is precisely where that discrimination misfires.

**Conclusion: the `heavy` numbers here measure the test clip, not the model.**
They must not be used to reject `heavy`. This is exactly the failure mode
`AGENTS.md` records from the face engine — a model that scored well on one
source and collapsed on another, because the benchmark had a signature of its
own.

The `lite` versus `full` comparison stands, because both handled the clip with
100% detection and high confidence throughout.

---

## Measurement 2 — knee angle against known ground truth

The squat clip's geometry is known exactly, so the true knee angle is computable
rather than estimated. This is a stronger test than any real footage could give
without a marker system.

| Model | mean error | median | p95 | valid frames |
|---|---|---|---|---|
| lite | **2.6°** | 2.1° | 6.6° | 58 / 290 |
| full | 3.5° | 3.3° | 6.9° | 31 / 290 |
| heavy | — | — | — | **0** |

Two things to read carefully here.

**The error is small — 2-3° — but the sample is tiny.** Only 58 and 31 frames of
290 produced a usable angle; the rest were refused because landmark confidence
fell below threshold during the deeper part of the squat. **A low error measured
on the easy 10-20% of frames is not the same as a low error overall**, and it
would be dishonest to present 2.6° as this model's accuracy.

What it does show: *when the model is confident, it is accurate*. The confidence
gate is doing real work — it declines rather than guessing. That is the
behaviour the framing gate was built for.

**These figures are far better than the 5-15° published range** from
Round 1, and that gap is itself evidence that the synthetic clip is easier than
reality: a rendered figure has hard edges, perfect contrast and no clothing
folds. **Trust the published 5-15° for design purposes, not the 2.6° here.**

---

## Measurement 3 — speed under motion

Same three models, replaying the squat clip:

| Model | fps | inference (median) | p95 | detection rate |
|---|---|---|---|---|
| lite | **53.1** | 11.8 ms | 13.5 ms | 99.7% |
| full | 39.1 | 17.5 ms | 21.3 ms | 95.2% |
| heavy | 15.1 | 57.3 ms | 61.1 ms | **100%** |

Note the reversal from the still clip: under motion `heavy` detected the subject
in **every** frame while `full` dropped to 95.2%. `heavy` is more robust at
finding a body and more sceptical about individual landmarks — both consistent
with it being the stronger model.

**On the original 14.8 fps complaint.** The simulator runs `full`, which
executes in 17.5 ms — about 57 fps of inference capacity. The 14.8 fps observed
live was therefore **never model-bound**; it was camera capture, colour
conversion and OpenCV window drawing. Round 1 predicted this from Google's own
25-27 ms figures, and it is now confirmed on this machine.

**Switching model to gain speed would have been the wrong fix.**

---

## Measurement 4 — centre of mass versus single joints

Round 1's claim: CoM is far steadier because mass-weighted averaging cancels
independent per-keypoint noise (published: 2.1 cm RMSE, r = 0.999, against a
documented ~11% flight-time bias).

Measured here on the still clip with `full`, using the Dempster (1955) segment
mass model — pixel standard deviation, lower is steadier:

| Quantity | pixel std |
|---|---|
| left hip (single landmark) | 1.41 |
| right shoulder | 1.08 |
| left pinky | 1.01 |
| left wrist | 0.90 |
| left ear | 0.55 |

The noisiest landmarks are hips and shoulders; the steadiest are on the head.
That is the opposite of the usual expectation that extremities are worst — and
it is likely another artefact of the rendered figure, whose torso is a flat
polygon with little internal texture to lock onto.

**This measurement is inconclusive and is reported as such.** The CoM-versus-joint
comparison needs real footage. The underlying argument from Round 1 — that
averaging over many mass-weighted landmarks cancels independent noise — is sound
mathematics and does not depend on this measurement, but the *size* of the
benefit is unquantified here.

---

## What Round 2 establishes

| Question | Answer | Confidence |
|---|---|---|
| Is 14.8 fps a model problem? | **No.** Inference is 17.5 ms ≈ 57 fps. The bottleneck is capture and drawing. | High — measured directly |
| Should we switch off MediaPipe `full`? | **Not for speed.** It is three times steadier than `lite` and fast enough. | High for lite/full |
| Is `heavy` worth using? | **Unresolved.** It scored badly here for a reason that is an artefact. It detected 100% under motion. | Low — needs real footage |
| Are angles stable enough for thresholds? | Yes at 0.5-1° on synthetic input; assume 5-15° in reality per Round 1. | Design for the pessimistic figure |
| Is CoM steadier than single joints? | **Not established here.** Argument is sound; measurement is inconclusive. | Low |

---

## What must be measured before any of this is acted on

1. **The same four measurements on real camera footage.** Everything above
   carries the synthetic clip's signature. The `heavy` result proves that
   signature is strong enough to invert a conclusion.
2. **The noise floor of a real still person**, which sets the hysteresis
   threshold width. The synthetic 0.51° is certainly optimistic.
3. **RTMPose via ONNX**, which was downloaded and prepared but not benchmarked
   this round: comparing it against MediaPipe on a rendered figure would repeat
   the `heavy` mistake, since RTMPose is likewise trained on photographs.

The camera needs an administrator-level device reset, or a reboot, before any of
this is possible. **The new camera you mentioned would resolve it too.**

---

## Round 3 will cover

- the recommendation, separating what is measured from what is inferred
- hysteresis threshold design sized to the pessimistic 5-15° figure
- centre-of-mass jump detection using the Dempster model
- the phone-portability of whichever backend is chosen
