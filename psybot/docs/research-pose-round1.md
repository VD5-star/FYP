# Round 1 — surveying the field

Research into body-tracking methods for jump detection, exercise correctness and
posture judgement. **No code was changed during this round.**

Deciding criterion, as set by the user: **joint-angle accuracy and temporal
stability**, not raw speed. A knee angle that jitters ±15° makes "is the knee
correctly bent" unanswerable no matter how fast the model runs.

---

## The three findings that change the plan

### 1. Google's own documentation says MediaPipe's depth is not good enough for joint angles

This is the most important thing found, and it contradicts almost every
MediaPipe fitness tutorial online.

MediaPipe's `pose_classification.md`, in its Future Work section, states that
Google is still improving BlazePose's Z prediction, and that doing so:

> "will allow us to use joint angles in the feature vectors, which are more
> natural and easier to configure"

Read carefully: Google is saying that **because Z is not yet good enough, they
cannot use joint angles**, and their shipped fitness sample instead falls back
to pairwise 2D distances.

**Consequence for us:** the simulator currently computes angles from 2D image
coordinates, which — by luck rather than judgement — is the correct choice. The
rule to keep: treat MediaPipe as a strong **2D** detector, compute angles in the
image plane, and constrain camera placement so the joint of interest lies in
that plane. Never use `z` or world-landmark depth for an angle.

### 2. No model reaches the accuracy the games were imagined to need

Peer-reviewed validation, markerless single-camera against marker-based lab
systems:

| Study | Setup | Joint angle error |
|---|---|---|
| Aleksic et al., *Sensors* 2024, 24(20):6624 | RTMPose, 2× iPhone 13 @240 fps, Qualisys reference, 12 subjects | knee **RMSE 6.9°**, LoA −6.7° to +11.9°; hip RMSE 8.0°; ankle 5.4° |
| Same paper, discrete events | knee flexion at transition | 105.2° marker vs 96.8° markerless — **8.4° systematic underestimate** |
| OpenCap / OpenPose, cited within | vertical jumps | **RMSE 11.6°–14.7°** |
| IEEE 9973559 | MediaPipe, rehab exercises | **RMSE 7.70°** |

And that 6.9° was achieved in a **laboratory**: two calibrated cameras, 240 fps,
controlled lighting, tight dark clothing, subject perpendicular at a fixed 2 m,
tripod height chosen to minimise perspective error.

Our app will run on a propped-up phone in a bedroom.

**Conclusion, stated plainly: expect 5–15° of joint-angle error, and no model
choice in this report will get us to ±2°.** The jitter reported from watching
the simulator is not a defect in our code — it is inside the published range for
this entire class of method.

This has to change the design of the games rather than the choice of model.

### 3. The fix for jitter is hysteresis, not a better model

Google's own rep-counting method, from the same document:

> "To avoid cases when the probability fluctuates around the threshold ...
> causing phantom counts, the threshold used to detect when the state is exited
> is actually slightly lower than the one used to detect when the state is
> entered. It creates an interval where the pose class and the counter can't be
> changed."

That is a Schmitt trigger. And it is the direct answer to our problem: **a
state machine does not need a stable absolute angle.** If the knee angle
jitters ±15°, set the enter and exit thresholds more than 30° apart and the
jitter cannot produce a false transition.

"Did they squat low enough" is answerable. "Your knee is at 87°" is not.

---

## Candidate models

### MediaPipe Pose Landmarker (what we use now)

Official accuracy, on Google's internal Yoga / Dance / HIIT sets, 17 COCO
keypoints:

| Model | Yoga mAP | Dance mAP | HIIT mAP | Pixel 3 GPU | MacBook Pro 2017 |
|---|---|---|---|---|---|
| Heavy | 68.1 | 73.0 | 74.0 | 53 ms | 38 ms |
| **Full** (ours) | 62.6 | 67.4 | 68.0 | 25 ms | 27 ms |
| Lite | 45.0 | 53.6 | 53.8 | 20 ms | 25 ms |

**Lite → Full is +17.6 mAP on Yoga.** Lite is materially worse at exactly the
non-standard poses an exercise game consists of. It should not be used for form
judging.

Full → Heavy costs ~11 ms on desktop for +5.5 mAP — nearly free for development
work.

Caveat worth recording: these numbers are on **internal, unpublished** datasets
that no third party can reproduce or audit. They are the most-cited MediaPipe
accuracy figures and they are not independently verifiable.

Architecture (Bazarevsky et al., arXiv:2006.10204 — a 4-page workshop paper, not
a full conference paper): heatmaps supervise training, then the heatmap head is
**discarded at inference**, leaving direct coordinate regression. Fast, but
weaker sub-pixel precision than heatmap methods.

**A structural jitter source:** the detector runs only on the first frame; after
that the region of interest is derived from the *previous frame's* landmarks.
A bad frame perturbs the next frame's crop, which perturbs the next estimate.
This feedback is independent of any smoothing applied downstream.

### RTMPose (OpenMMLab) — the strongest candidate

Jiang et al., arXiv:2303.07399. COCO val, 256×192 input:

| Model | AP (COCO) | Params (M) | ONNX i7-11700 | Snapdragon 865 ncnn |
|---|---|---|---|---|
| RTMPose-t | 68.5 | 3.34 | 3.20 ms | 9.02 ms |
| RTMPose-s | 72.2 | 5.47 | 4.48 ms | 13.89 ms |
| **RTMPose-m** | **75.8** | 13.59 | **11.06 ms** | **26.44 ms** |
| RTMPose-l | 76.5 | 27.66 | 18.85 ms | 45.37 ms |

11 ms on a CPU is roughly **90 fps** — against our current 14.8 fps.

- **Apache-2.0.** No licensing hazard.
- Pre-exported ONNX models downloadable directly; ONNXRuntime, ncnn, OpenVINO,
  TensorRT backends; **Android ncnn example provided**.
- Ships "pure Python inference without MMDeploy, MMCV" examples — which removes
  the usual objection to OpenMMLab's heavy dependency chain.
- It is the model used in the peer-reviewed CMJ validation above, so choosing it
  means inheriting a **citable error characterisation** — valuable for the
  report.

Cost: it is **top-down**, so it needs a person detector upstream (RTMDet-nano,
0.99M params). Full pipeline ≈ 26.6 ms on the same CPU.

### YOLO-pose (Ultralytics) — rejected on licensing

Accuracy is genuinely strong (YOLO26m-pose: 68.8 mAP). The problem is the
licence.

From Ultralytics' own licensing page: using their code, models, architectures or
**trained models** requires either open-sourcing **the entire project** under
AGPL-3.0, or a paid Enterprise licence. This applies *even if you train from
scratch, use no pretrained weights, or use it only internally*. Trained models
are AGPL by default.

University coursework is listed as acceptable under AGPL. But this is a
**mental-health application intended to run on a phone**. AGPL-3.0 would require
publishing the complete source of that application. For an app handling
mental-health conversations, that is a privacy consideration as well as a
commercial one.

**Rejected.** RTMPose matches or beats it under Apache-2.0. There is no accuracy
gain here worth the liability.

### MoveNet (Google) — not competitive for this task

Lightning (192²) and Thunder (256²), 17 keypoints, bottom-up heatmap. Fast in
browsers (Thunder 77 fps on a 2019 MacBook, WebGL).

Two disqualifying points:

1. **No published accuracy numbers in the announcement.** The TensorFlow blog
   post contains zero quantitative accuracy figures — the accuracy claim is a
   testimonial from a commercial partner's CEO. That is marketing, not
   measurement.
2. **Effectively frozen since 2021.** No new generation; Google's edge-vision
   work has consolidated into MediaPipe Tasks / LiteRT.

It does have one point worth stealing: it ships built-in temporal filtering,
described as tuned "to simultaneously suppress high-frequency noise (i.e.
jitter) and outliers ... while maintaining high-bandwidth throughput during
quick motions." That is a One-Euro-family filter, which is what we already use.

### 3D lifting (MotionBERT, VideoPose3D, MHFormer) — wrong tool

| Method | MPJPE on Human3.6M |
|---|---|
| MotionBERT (ICCV 2023) | **37.2 mm** finetuned |
| VideoPose3D (T=243) | 48.1 mm |
| Martinez et al. 2017 | 54.6 mm |

Three reasons to reject for a live in-chat game:

1. These are **2D-lifting** methods. The pipeline is RGB → 2D detector → lifter,
   so errors compound, and the headline numbers assume high-quality 2D input.
2. **37 mm is not as good as it sounds for angle work.** MPJPE is measured after
   hip-alignment. A 37 mm error at the knee, on a ~400 mm shank, is roughly
   **5° of angular error** before adding the hip and ankle errors that also
   define that angle — on a clean lab dataset.
3. They need a **temporal window of up to 243 frames — about 8 seconds**. That
   latency is incompatible with a responsive game.

Worth revisiting later as an *offline* check on our 2D approach, not as the
live path.

---

## Smoothing: the trap we were heading into

SmoothNet (Zeng et al., ECCV 2022, arXiv:2112.13715) ran the controlled
experiment that explains the jitter, and it contains the single most useful
table found this round.

First, the diagnosis:

> "For single-frame models, we observe that the position errors decrease, but
> the acceleration errors become larger as the epochs increase, indicating that
> the single-frame methods which extract only spatial information are likely to
> **sacrifice smoothness in exchange for localization performance**."

So MediaPipe is *trained* in a way that trades temporal smoothness for
per-frame accuracy. The jitter is structural.

Now the trap. Human3.6M, 3D pose, **Accel is a jitter proxy**:

| Method | Accel | MPJPE (mm) |
|---|---|---|
| FCN, raw | 19.17 | 54.55 |
| + One-Euro, moderate | 3.80 | 55.20 |
| **+ One-Euro, aggressive** | **0.94** | **143.24** |
| + SmoothNet | 1.03 | **52.72** |

**Tuning One-Euro hard enough to kill the jitter — a 20× reduction — made the
pose 2.6× worse.** This is exactly the mistake we would have made by increasing
the filter strength in response to a shaky-looking skeleton.

SmoothNet achieves the same jitter reduction *while improving* accuracy, at
0.03M parameters. Typical reductions across estimators: **85–87%**.

Also relevant: Savitzky-Golay outperformed One-Euro in these benchmarks, but it
is **non-causal** — it needs future frames. Usable for analysing a recorded
session, not for live feedback.

The One-Euro authors' own tuning procedure (Casiez et al., CHI 2012) is worth
following exactly: hold still and lower `min_cutoff` until jitter is acceptable;
then move fast and raise `beta` until lag is acceptable. Our current constants
were guesses and are still marked provisional in the code.

---

## Jump detection: measure the centre of mass, not the flight time

From the same Aleksic validation:

| Measurement | Result |
|---|---|
| **CoM vertical trajectory** | RMSE **0.021 m**, bias 0.000, r = **0.999** |
| Flight time | **~60 ms overestimate** (0.55 s → 0.61 s), p < 0.01 |
| Jump height | ~2 cm overestimate, p < 0.01 |
| Propulsive phase | 33 ms underestimate |

**Why CoM is so much better:** it is a mass-weighted average over many
keypoints, so independent per-keypoint jitter cancels. This is a structural
insight worth generalising — **derive metrics from an aggregate of many joints,
never from a single joint.**

Flight time is the worst choice available: toe-off and touchdown are the hardest
events for a keypoint detector (motion blur, foot occlusion, an ill-defined toe
point), and since `h ∝ t²`, an 11% timing error becomes a ~23% height error.

CoM is computed with the **Dempster (1955) segment mass model** — foot 1.5%,
lower leg 4.65%, upper leg 10%, upper body 43%, with segment centres at 43.3%
along the limb from the distal joint. A standard, citable recipe that is
straightforward to implement from the landmarks we already have.

**A validation dataset exists:** BioCV (Evans et al., *Scientific Data* 11:1300,
2024). 15 participants, synchronised 200 Hz video + Qualisys mocap + **Kistler
force plates**, including 10 countermovement jumps each, plus markerless trials.
CC-BY, on request from the University of Bath. If the report needs a defensible
accuracy number for jump detection, this is the way to get one.

Note on a commonly cited comparison: the *My Jump* smartphone app reports
r > 0.95 against force plates **but overestimates by 4.32 cm on average**. The
same pattern recurs everywhere — excellent correlation, meaningful absolute
bias. **A paper reporting only `r` is hiding this.**

---

## What this round changes

| Belief before | After |
|---|---|
| A better model will fix the jitter | No. 5–15° error is inherent to single-camera markerless pose. |
| Joint angles are the natural feature | Only in 2D, and only coarsely. Google avoids them entirely. |
| Stronger smoothing will steady the skeleton | Over-smoothing made pose 2.6× worse in a controlled test. |
| Jump height from flight time | CoM displacement: 2.1 cm RMSE vs a documented 11% flight-time bias. |
| MediaPipe is the obvious choice | RTMPose-m is more accurate *and* ~6× faster on CPU, under Apache-2.0. |

**The uncomfortable conclusion, stated for the record:** the games must be
designed around coarse, hysteretic, multi-joint judgements — "you got low
enough", "you jumped" — not fine angular ones. For a mental-health application,
avoiding any implication of clinical-grade movement assessment is worth doing
deliberately, and it happens to be what the measurements support.

---

## Next: Round 2 measures these claims on this laptop

Everything above is other people's numbers on other people's hardware. That is
exactly what `AGENTS.md` warns against, and exactly how the face engine was
misled before.

Round 2 will measure, on this machine and this camera:

1. MediaPipe lite vs full vs heavy — fps and joint-angle noise floor
2. RTMPose via ONNX — whether the 11 ms claim holds here
3. The **noise floor of a deliberately motionless subject**, which is the number
   that decides whether hysteresis thresholds are viable
4. Whether CoM is measurably steadier than a single joint, as claimed
