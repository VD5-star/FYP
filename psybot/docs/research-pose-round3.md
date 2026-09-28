# Round 3 — recommendation

Rounds 1 and 2 established what is possible. This round decides what to build.
**No production code was changed in any of the three rounds.**

Everything here separates **measured** from **inferred**, because the whole
value of three rounds of research is lost if they get mixed.

---

## The recommendation in one page

| Decision | Answer | Basis |
|---|---|---|
| Tracking model | **Keep MediaPipe Pose `full`** | Measured here: 3× steadier than `lite`, 17.5 ms ≈ 57 fps capacity |
| Fix the 14.8 fps | **Not a model problem** | Measured: inference is 17.5 ms; the cost is capture + drawing |
| Joint angles from `z` | **Never** | Google's own docs; independently, ML Kit weights z at 0.2 |
| Detect states | **Hysteresis, dead zone ≥ 40°** | Two independent lines converge (below) |
| Detect jumps | **Centre of mass, not angles, not flight time** | Published: 21 mm vs 5–9°; flight time biased +61 ms |
| Judge form | **Range-over-rep, not instantaneous** | Published: 100% good-form, 80% bad-form, safe failure direction |
| Highest-value feature | **Per-user calibration** | Removes all five documented systematic biases at once |
| Report jump height in cm | **No** | ±18% at 40 fps; report a relative score |

---

## 1. Why the model stays MediaPipe `full`

Round 2 measured, on this laptop, replaying identical input:

| Model | inference | angle std (still) | detection under motion |
|---|---|---|---|
| lite | 11.8 ms | 1.52° | 99.7% |
| **full** | **17.5 ms** | **0.51°** | 95.2% |
| heavy | 57.3 ms | *artefact* | 100% |

`full` is three times steadier than `lite` at a cost of 6 ms. Since stability is
the deciding criterion, `lite` is rejected despite being faster.

**RTMPose was investigated and prepared but not adopted.** Round 1 found it
genuinely attractive — 75.8 AP against MediaPipe's ~62-68, 11 ms on CPU,
Apache-2.0, a peer-reviewed error characterisation, and an Android ncnn path.

It is not adopted now for one reason: **benchmarking it on the synthetic clip
would repeat the exact mistake Round 2 caught.** `heavy` scored sixteen times
worse than `full` purely because it correctly refused to trust the legs of a
rendered figure. RTMPose is likewise trained on photographs. Any comparison on
that clip would measure the clip, not the models.

RTMPose remains the leading candidate if real footage shows MediaPipe's
stability is insufficient. **That comparison needs the camera.**

---

## 2. The hysteresis design, with numbers

This is the core of the answer to "the tracking sometimes sticks and doesn't
work". The fix is not a better model — it is to stop asking the model for
precision it does not have.

### Why the dead zone must be at least 40°

Two entirely independent lines of evidence converge:

**Line 1 — measured error.** Aleksic et al. 2024, RTMPose against Qualisys,
12 subjects: knee angle limits of agreement **−6.7° to +11.9°**, hip **−6.1° to
+14.0°**. Any dead zone narrower than about 20° sits inside the noise.

**Line 2 — Google's shipped constant.** ML Kit's `RepetitionCounter` uses
enter 6/10, exit 4/10 — a dead zone of **20% of the signal's range**. A squat's
interior knee angle spans roughly 170° standing to 80° at the bottom, so 20% of
that range is **18°**.

Two unrelated arguments landing on ~18-20° is the strongest evidence available.
Doubling it for safety in uncontrolled conditions gives the recommended values:

| Exercise | Signal | Enter | Exit | Dead zone |
|---|---|---|---|---|
| Squat | interior knee angle | < 100° | > 160° | **60°** |
| Push-up | interior elbow angle | < 100° | > 155° | **55°** |
| Arm raise | shoulder-torso angle | > 140° | < 60° | **80°** |

Cross-check: hobby projects that reportedly work in practice use 40-70°.
Projects using narrow bands do not get reported, because they do not work.

### Three guards on top of the threshold

Hysteresis alone is not enough. From shipped implementations:

1. **N-frame confirmation** — the condition must hold 3-5 consecutive frames
   (75-125 ms at 40 fps) before the state changes. Kills single-frame spikes
   that hysteresis cannot.
2. **Minimum dwell time** — reject state changes faster than 300 ms. A human
   squat bottoms out for longer; a two-frame flicker is noise.
3. **Sequence validation** — a rep requires the full `down → bottom → up`
   sequence, not a single edge crossing.

Point 3 is the most valuable and the least obvious. LearnOpenCV's implementation
counts a rep only on the exact sequence `[s2, s3, s2]`, and — importantly —
counts a partial squat (`[s2]` alone) as an **improper rep** rather than
ignoring it. That distinguishes "didn't go deep enough" from "didn't move",
which is the difference between useful feedback and silence.

### A trap worth recording

LearnOpenCV's published thresholds are **angles to the vertical**, not interior
joint angles. Their "knee angle 95°" means the thigh is roughly horizontal — a
deep squat. Copying those numbers into an interior-angle system would invert the
meaning. The numbers in the table above are interior angles.

---

## 3. Jump detection: measure the centre of mass

The claim from Round 1 is now backed by the full validation data:

| Quantity | Bias | Limits of agreement | RMSE |
|---|---|---|---|
| **CoM vertical position** | 0.000 m | ±0.035 m | **21 mm** |
| Toe vertical position | −21 mm | −73 to +32 mm | 35 mm |
| Knee angle | +2.6° | −6.7 to +11.9° | 6.9° |
| Hip angle | +3.9° | −6.1 to +14.0° | 8.0° |

**Position beats angle, decisively.** CoM correlates at r = 0.999 against a
laboratory system; angles at 0.98-0.99 with far wider limits. The reason is
structural: CoM is a mass-weighted average over many landmarks, so independent
per-keypoint noise cancels.

Round 2 implemented the Dempster (1955) segment model in the benchmark harness
and verified it computes a sensible CoM, so the recipe is ready.

### Do not use flight time

Flight time is biased **+61 ms** (0.547 s measured against 0.608 s markerless,
p < 0.01), which propagates to a +2.2 cm jump-height bias. And since
`h ∝ T²`, the error squares.

At 40 fps the frame period is 25 ms, so a ±1 frame error at each of take-off and
landing is ±50 ms ≈ **±9% on flight time ≈ ±18% on height**. The validation
study resampled from 240 fps and still saw the bias.

**Conclusion: detecting that a jump happened is easy. Measuring how high is
not.** Report a relative score, never a number in centimetres.

### Distinguishing a jump from standing on tiptoes

The literature barely covers this, so it is reasoned from physics — flagged as
**inferred, not measured**:

1. **Free-fall signature (strongest).** During flight, vertical acceleration is
   −g and velocity crosses zero exactly once. Fit a parabola near the apex and
   check the fitted acceleration against g; reject if it deviates by more than
   ~30%. Tiptoes has no free-fall phase at all.
2. **Ankle-hip decoupling.** In a jump, ankle and hip rise together. On tiptoes
   the hip rises while the **ankle stays put**.
3. **Magnitude.** Tiptoe rise is 5-12 cm; a bilateral jump 20-45 cm. Supporting
   evidence only — it overlaps with weak one-legged jumps.
4. **Duration.** Require at least 8-10 frames of flight.

### Camera shake

Every validation study bolts the phone to a tripod, so the literature is silent.
Our users will prop a phone against a book.

Mitigation, **inferred**: camera translation moves *all* landmarks identically,
so the median displacement across all 33 landmarks estimates global motion and
can be subtracted. This fails during flight, when the body genuinely does
translate — so estimate global motion from the pre-take-off and post-landing
windows and interpolate across the gap.

---

## 4. What actually causes "it sometimes sticks"

Round 2 proved the frame rate was never model-bound. The remaining causes are
identifiable and each has a shipped fix:

| Cause | Fix | Source |
|---|---|---|
| Subject turned away from camera | Compute nose-to-shoulders offset angle; above 35°, **refuse to analyse** and say so | LearnOpenCV |
| Landmark spikes and left/right swaps | Median filter, width 5, applied twice | Pose Trainer |
| Stale state after dropped frames | Clear all filter memory on any gap > 100 ms | ML Kit |
| Another person enters frame | Torso length or hip centre jumping > 20% in one frame is physically impossible — reset and discard the rep | inferred |
| Low-confidence landmarks | Gate on **smoothed** visibility; MediaPipe itself low-passes visibility at α = 0.1 because it is noisy | MediaPipe config |

The `RESET_THRESHOLD_MS = 100` rule deserves emphasis. At 40 fps a 100 ms gap
means four dropped frames; blending across it produces garbage velocity. Our
`PoseSmoother` already resets when the subject is lost, but **not on a timing
gap** — that is a real defect to fix.

### A discovery that changes the smoothing plan

MediaPipe already applies a One-Euro filter internally, and the production
constants are public:

```
screen landmarks : min_cutoff 0.05, beta 80.0
world landmarks  : min_cutoff 0.1,  beta 40.0
visibility       : low_pass alpha 0.1
```

Google's own comments translate these: **α ≈ 0.01 when static, α ≈ 0.94 when
moving fast.**

Two consequences:

1. **Our `PoseSmoother` is a second filter stacked on a first.** That adds lag
   without adding much stability, and lag is exactly what "the tracking is
   laggy" describes. Our constants (`min_cutoff 1.2, beta 0.02`) are far more
   aggressive than Google's.
2. **α ≈ 0.94 during fast motion means MediaPipe deliberately passes raw noise
   through when moving quickly**, to avoid lag. So the input is *not* clean
   during a jump — precisely when it matters.

This was not visible in Round 2's measurements and is the single most actionable
finding of Round 3.

---

## 5. Judging form: what is honest to promise

The one study that compared approaches on the same data (Pose Trainer, 100+
videos, 4 exercises):

| Approach | Result |
|---|---|
| Geometric rules on **range-over-rep** | **100%** of good form correct, **80%** of bad form flagged |
| DTW + 1-nearest-neighbour | F1 0.73-0.85, but on 7-15 test samples |

Rules win, and the failure direction is the safe one: **zero false positives on
good form.** Never wrongly telling a correct user they are wrong matters more
than catching every error — especially in a mental-health context.

The key insight: features are the **range, minimum and maximum of an angle over
a whole repetition**, never an instantaneous value. A systematic +3.9° hip bias
cancels entirely in a range computation. This is why per-user calibration is the
highest-value feature available — it converts every absolute threshold into a
relative one and removes all five documented biases at once.

### What must not be promised

ECCV 2022 state of the art (Parmar et al., Fitness-AQA) states directly:

> "off-the-shelf pose estimators struggle to perform well on the videos recorded
> in gym scenarios due to factors such as camera angles, occlusion, illumination
> and clothing. To aggravate the problem, the errors to be detected in the
> workouts are very subtle."

They found learned representations outperform both 2D **and** 3D off-the-shelf
pose estimators for error detection.

**Coarse judgements are achievable: depth reached, gross alignment, range of
motion. Subtle ones are not: knee valgus, lumbar rounding.** For a
mental-health application this aligns with `AGENTS.md`'s existing boundary —
the app is not a clinical instrument, and movement assessment should not imply
that it is.

---

## 6. Expectation setting, stated plainly

Aderinola et al. discarded **19 of 96 unilateral jumps (20%)** as failure cases —
in a laboratory, with cooperative subjects, controlled clothing and a tripod.

Our users will be in bedrooms with a propped-up phone.

**Build an explicit reject path.** "I couldn't read that clearly" is a feature,
not a defect, and it is far better than a confidently wrong rep count. This is
the same principle already built into the framing gate, which declines to
produce numbers when the torso is unreadable.

---

## 7. What is measured, and what is not

Being explicit, because the value of this research depends on the distinction.

### Measured on this machine (Round 2)
- MediaPipe lite / full / heavy: inference time, detection rate, angle noise
- That the 14.8 fps was never model-bound
- That `full` is 3× steadier than `lite`
- That the Dempster CoM computation is correct

### Measured by others, peer-reviewed (Rounds 1 and 3)
- Joint angle error 5-15° in real conditions
- CoM position error 21 mm, r = 0.999
- Flight time bias +61 ms, jump height bias +2.2 cm
- Rule-based form: 100% / 80%
- 20% failure rate on unilateral jumps in a lab

### Vendor production constants (verified in source)
- MediaPipe's internal One-Euro constants
- ML Kit's 20%-of-range dead zone, 100 ms reset, z-weight 0.2

### Inferred, never measured — treat as hypotheses
- Jump versus tiptoe discrimination by parabola fit
- Camera-shake correction by median landmark displacement
- Subject-switch detection by torso-length discontinuity
- That the recommended thresholds suit *our* users

### Not established at all
- Whether RTMPose is better here
- Whether CoM is measurably steadier than single joints **on real footage**
- The noise floor of a real still person on a real camera

---

## 8. What to do when the camera works

In priority order:

1. **Re-measure the Round 2 numbers on real footage.** The `heavy` result proves
   a synthetic clip can invert a conclusion.
2. **Measure the noise floor of a real still person.** This sizes the hysteresis
   dead zones and is the number every threshold depends on.
3. **Reconsider our `PoseSmoother`.** MediaPipe already filters; ours may be
   adding lag for little gain. Measure with it disabled.
4. **Add the 100 ms filter-reset guard.** A clear defect regardless of anything
   else.
5. **Then** benchmark RTMPose against MediaPipe on that real footage.

The new camera will resolve the access problem. Until then, every number in
Round 2 carries the synthetic clip's signature and must be treated as
provisional.
