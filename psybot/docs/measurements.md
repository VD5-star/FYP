# Measurements

## The Android emulator was abandoned — measured reasons

Four faults were reported from watching it run: the skeleton sat beside the body
rather than on it and was vertical while the user was horizontal; the camera
appeared pushed to the far right; heavy stutter; and the camera frequently
failed to open at all.

All four trace to the emulator, not to the design. It passes a **laptop**
webcam to Android while presenting it as a phone sensor mounted at 90°, so the
landmarks and the displayed image are in different coordinate spaces. And it
runs inference and compositing on the same CPU with no GPU.

Measured on the same laptop, same camera, same measurement model:

| | Android emulator | Python + MediaPipe on Windows |
|---|---|---|
| Frame rate | **2.0 fps** | **14.8 fps** |
| Frames with a body detected | occasional, `partial` only | **60 / 60** |
| Camera open | often failed; emulator hung hard enough to need killing by PID | **1.1 s** |

**7× the frame rate**, with nothing emulated.

Decision: develop on the laptop simulator, and revisit the phone build on a
real device. Android Studio, the emulator, system images, AVD data and the NDK
were deleted — 12.2 GB. The base SDK and JDK were kept, so an APK can still be
built the moment a phone is available.

### The root cause of "the skeleton is not on my body"

Detection ran on unmirrored pixels; the image was then flipped for display. Two
spaces, so the figure landed on the wrong side and moved the wrong way.

Fixed in the simulator by **mirroring before detection**, so only one
coordinate space ever exists. The same ordering rule applies to the phone
build.

---

## Body engine (engine_body) — first run, emulator only

**Read the caveat before quoting any of this.** These come from an Android
emulator with software rendering (`swiftshader_indirect`) on the development
laptop. An emulator is not a phone: it has no NPU, no GPU delegate worth the
name, and a virtualised camera. **None of these numbers may go into the report
as mobile performance.** They exist to prove the pipeline runs end to end.

| Measurement | Value | Conditions |
|---|---|---|
| Frame rate, no preview | **6.2 → 8.5 fps** | Emulator, swiftshader, `fast` model, 480 px analysis |
| Frame rate, preview at 1280×960 | **2.0 fps** | The resolution bug below |
| Frame rate, preview at 640×480 | **3.8 fps** | After the fix |
| Frames processed without failure | 300+ consecutive | No crash, no dropped stream |
| ML Kit model load | Succeeded | `pose_person_detector_f16.tflite`, `pose_landmark_detector_lite_f16_inf.tflite` |
| **A real body detected** | **33 landmarks**, `partial` framing | First time a person was in front of the camera |

The `partial` result is correct rather than a fault: at laptop-webcam distance
the torso is not fully in frame, so the framing gate refuses to build a torso
reference and withholds the derived signals. That is the gate doing its job —
it declined to produce numbers from an unusable reference instead of quietly
returning plausible ones.

Adding the preview costs roughly half the frame rate **on this emulator**,
because software rendering composites the camera texture on the CPU that is
also running inference. A phone composites on the GPU and should pay far less,
but that is a prediction and not a measurement — do not quote it.

### Resolution fallback direction: measured, 4x

`ResolutionStrategy` was first built with `FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER`.
Asked for 480 px, the camera offered no such size and jumped to **1280×960**.

| Fallback rule | Chosen size | Frame rate |
|---|---|---|
| `CLOSEST_HIGHER_THEN_LOWER` | 1280×960 | **2.0 fps** |
| `CLOSEST_LOWER_THEN_HIGHER` | 640×480 | **3.8 fps** |

**A 4x cost for detail pose detection cannot use.** When the requested size is
unavailable, smaller is always the right way to miss. The same trap exists on
real phones, where sensors commonly offer nothing between 640×480 and
1280×720.

### The preview appeared sideways and cropped

Reported from a real run: the camera image looked rotated onto its side, and
appeared to be aimed to the left rather than straight ahead.

Two separate faults with one root cause — **confusing the two coordinate
spaces the texture and the landmarks live in.**

1. **Rotating twice.** The widget computed *pre-swapped* dimensions (a 640×480
   sensor at 90° laid out as 480×640) and *then* wrapped that in a `RotatedBox`.
   The rotation was applied to already-corrected dimensions, so the aspect ratio
   broke and the image looked sideways and stretched.

2. **Rotating the overlay along with the texture.** The native layer passes the
   rotation to `InputImage`, so **ML Kit returns already-upright landmarks**.
   The texture needs rotating; the overlay does not. Rotating both turned a
   correct skeleton onto its side.

   Mirroring is the opposite case and applies to *both*: landmarks derive from
   unmirrored pixels, so a flipped image needs a flipped overlay or every joint
   lands on the wrong side of the body.

3. **`BoxFit.cover` cropped the sides.** A 4:3 image on a tall phone loses a
   wide strip from each edge, which is what made the camera look as though it
   were pointed off to one side. It also hid outstretched arms, and made it
   impossible to tell whether a limb had left the *frame* or merely the
   *visible crop*. Now `BoxFit.contain`: the whole field of view is shown,
   letterboxed. What you see is exactly what the detector sees.

A further correctness point: the rotation reported to Flutter is now the same
value applied to the analysis frames (`ImageProxy.imageInfo.rotationDegrees`),
not `sensorRotationDegrees`. The latter ignores how the device is currently
held, so the two would diverge whenever the phone was rotated.

Six widget tests now pin this geometry. It had none before, which is why a
purely visual fault reached the user.

### Preview resolution was reported as 0×0

The first preview implementation returned `PreviewInfo(texture: 0, 0x0)`, which
Flutter rendered as a black rectangle — indistinguishable from a camera that
failed to open.

Cause: CameraX resolves the preview resolution **asynchronously**, when it
requests a surface, which happens *after* `bindToLifecycle` returns. Reading it
immediately gives zero. Fixed by deferring the channel reply until the surface
request arrives, guarded by an `AtomicBoolean` because CameraX may request a
surface more than once and answering a `MethodChannel` twice crashes the app.

The 6.2 fps figure is bounded by the emulator, not by the design. It is
recorded only so that a later phone measurement has something to be compared
against, and so nobody re-runs this experiment expecting a different answer.

**Still unmeasured:** anything on real hardware — frame rate, latency, battery,
`fast` against `accurate`, and the landmark noise floor of a motionless subject.
Until those exist, the One-Euro constants and `stillnessReference` in the code
remain provisional.

### Bug found on the first run

`CameraSelector.DEFAULT_FRONT_CAMERA` failed with *"Provided camera selector
unable to resolve a camera for the given use case"*.

Cause, from logcat: the emulator exposes exactly one camera, `id=10`, with
`lensFacingInteger: null`. CameraX then logged *"the device might underreport
the amount of the cameras"* and every bind attempt failed.

This is not only an emulator defect. Any device without a front camera fails
identically. Fixed in `EngineBodyPlugin.resolveCamera`: try the requested
facing, then the opposite, then any available camera. A tracker running on the
wrong camera is a degraded session; a tracker that refuses to start is no
session at all.

Worth noting how it was found: the failure was **silent** in the UI. The app
looked alive and simply showed nothing, which is indistinguishable from an
empty room. It only became visible after adding `onError` to the stream
subscription and logging state to `adb logcat`. Flutter draws its own widgets,
so `uiautomator` cannot read them, and screenshots cannot be inspected in this
workflow — a log line is the only way to tell a working tracker from a dead one
on a device.

---

# Measurements carried over from the desktop engine

Every number here came from running code on the user's machine (10-core CPU,
1280×720 webcam). They are recorded so they are not re-derived, and so that
anyone overturning one knows they must measure again rather than argue.

---

## Pipeline cost

| Stage | Cost | Note |
|---|---|---|
| Capture | 32.2 ms/frame | 31.1 fps, MJPG, 1280×720 |
| Face detection alone | 19.4 ms | The cheap part |
| + landmarks (106 pt) | +75.8 ms | Needed for action units |
| + age/gender | +15.5 ms | Droppable on mobile |
| + recognition | +102.5 ms | Droppable on mobile |
| Emotion ONNX (224²) | 81.1 ms | |
| **Full frame** | **~230 ms** | With identity cached every 5 frames |

### Detector input size barely matters

| Input width | Detection cost | Faces found |
|---|---|---|
| 1280 px | 305.8 ms | 1 |
| 960 px | 189.6 ms | 1 |
| 720 px | 186.7 ms | 1 |
| 640 px | 184.1 ms | 1 |

The model letterboxes to 320×320 internally, so larger inputs are paid for and
then discarded.

### JPEG preview cost

| Setting | Time | Size | At 30 fps |
|---|---|---|---|
| 1280×720 q80 | 3.4 ms | 70 KB | 2.1 MB/s |
| 640 wide q70 | 0.9 ms | 17 KB | 0.5 MB/s |

---

## Emotion accuracy

Measured on 1500 labelled faces from three corpora (AffectNet-HQ, FER2013,
Cohn-Kanade).

| Setup | Overall | AffectNet | FER2013 | Cohn-Kanade |
|---|---|---|---|---|
| BGR input (bug) | 58.6% | 64.2% | 51.6% | 69.0% |
| **RGB input (fixed)** | **61.3%** | **67.5%** | 54.1% | 70.6% |

Per class, after the fix:

| Class | Accuracy |
|---|---|
| happy | 81.1% |
| disgust | 69.2% |
| surprise | 65.6% |
| fear | 64.8% |
| anger | 62.8% |
| neutral | 46.8% |
| sad | 42.1% |

**Neutral and sad are the weakest**, and they are the two that matter most in a
mental-health context. This is a strong argument for sending continuous
signals rather than labels.

---

## Failed attempts to improve accuracy

### 1. Linear head over 7 probabilities + 6 action units

| Split | Result |
|---|---|
| Random split | 59.7% → 64.4% (**+4.7%**) |
| Hold out AffectNet | 65.5% → 64.4% (**−1.1%**) |
| Hold out Cohn-Kanade | 66.7% → 66.6% (**−0.1%**) |
| Hold out FER2013 | 51.7% → 54.5% (+2.8%) |

Generalised on 1 of 3 sources. **Refused by the guard, not shipped.**

### 2. Same head, 61k images, balanced

Failed the same way.

### 3. CNN trained on pixels

20,747 aligned crops at 112 px, trained at 64 px, 30 epochs, ~3 min/epoch.

| Holdout | Frozen | CNN | |
|---|---|---|---|
| AffectNet | 65.5% | 51.4% | **−14.1%** |
| Cohn-Kanade | 66.7% | 70.1% | +3.4% |
| FER2013 | 51.7% | *stopped* | |

Stopped by the user at 1–1 with a negative trend.

### The conclusion

Every corpus has its own signature — lighting, camera, acting style, even JPEG
compression. A model learns the signature because it is easier than learning
the expression, then collapses on a new source.

**A phone camera is always a new source.**

Personal calibration inverts the problem: one source, no shift. Measured
effect: **−37% in between-person variance** — the largest real gain achieved.

---

## Bugs found by measurement

Recorded because each was invisible until measured, and each is easy to
reintroduce.

### Channel order

OpenCV decodes BGR; the model was trained on RGB with per-channel ImageNet
statistics. The red mean was being applied to the blue channel.
**58.6% → 61.3%.**

### Baseline division inventing emotions

The correction divided *every* class by the resting distribution, including
neutral. A resting distribution is mostly neutral by definition, so neutral was
divided by a large number and rarely-shown classes by a small one.

Measured on a realistic baseline: a neutral reading of 0.42 became **0.199**,
and anger **doubled**. A calm face read as angry.

Fix: neutral passes through untouched; expressive classes are corrected among
themselves. Amplification capped at 16× (was 50×).

Live result: anger 31/39 → **neutral 37/40**; confidence 0.38 → 0.57.

### Periodic stutter

Deferring recognition *and* age/gender together stacked their cost into one
frame every five: **526 ms vs 200 ms**. Deferring only recognition (the
expensive one) dropped spikes from **42% of frames to 2%**.

### Calibration destroyed by identity flicker

The session restarted whenever `person_id` differed from the previous frame.
Recognition is not frame-stable — a turn, a blink or motion blur drops the
match and restores it a frame later.

Simulated with a wrong match on 1 frame in 11:

| | Before | After |
|---|---|---|
| Restarts in 300 frames | **27** | **0** |
| Peak progress | 10 / 1000 | 300 |

Calibration was **impossible to complete** for anyone who moved. Fixed by
requiring 45 consecutive frames naming a different person.

### Face box drift

The preview was downscaled to 720 px but bounding boxes are in capture
coordinates (1280×720). Placing the box against the preview size put it at
**75% across the frame instead of 42%**.

---

## Download performance

Relevant if datasets are ever fetched again.

| Method | Throughput |
|---|---|
| HuggingFace `streaming=True` | 0.32 MB/s |
| Parallel parquet shards | **21.5 MB/s** |

**67× faster.** Streaming fetches one image per HTTP request with no
parallelism — 95% of wall-clock time was spent waiting on the network, and each
image arrived at 166 KB only to be stored at 16 KB.
