# Tracking range and re-acquisition — measured

The reported problem: tracking is unreliable as the subject moves nearer,
further, or fast. This is the cause, measured, and the fix.

## The cause: a 6.7× gap between two stages

MediaPipe's pose pipeline has two stages with very different range.

The **detector** is face-proxy based. Google's own paper says so — *"we use a
fast on-device face detector as a proxy for a person detector"*, and *"the head
of the person should always be visible"*. It runs on the whole frame, so a
distant subject is a small face in a large image.

The **landmark model** never sees the whole frame. It sees a crop rescaled to
256×256, so the subject is always large in the tensor regardless of distance.

Measured on our own imagery, a real person rendered at decreasing size in a
1280×720 frame:

| | works down to | roughly |
|---|---|---|
| fresh detection | **400 px tall** | 2.5 m |
| established tracking | **60 px tall** | 17 m |

**A 6.7× gap.** And read from MediaPipe's graph definitions, the detector is
gated off entirely while tracking succeeds (`GateCalculator` with
`DISALLOW:prev_pose_rect_from_landmarks_is_present`), with **no periodic
re-detection anywhere in the pipeline**.

### The failure this produces

Reproduced exactly:

```
subject tracked at 240 px (~4 m)      ok
subject steps out of frame for 1 s    lock lost
subject returns at 240 px             CANNOT RE-ACQUIRE
subject walks back to 480 px (~2 m)   recovered
```

The user has to walk towards the camera to be seen again. Turning away, being
occluded, or moving too fast for the crop all break the lock the same way.

## The fix: search a crop, not the frame

When the lock is lost, hand the detector a **crop around where the subject last
was, upscaled**, instead of the whole frame.

| subject height | full frame | crop + upscale |
|---|---|---|
| 400 px | yes | yes |
| 340 px | **NO** | yes |
| 240 px | **NO** | yes |
| 140 px | **NO** | yes |
| 60 px | **NO** | yes |

Landmark error against a close-up reference stays flat at 0.06–0.08 torso units
across the whole range — the extra range is not bought with a worse pose.

End to end on the failing scenario:

```
before:  recovered  0/20 frames  (NEVER)
after:   recovered 14/20 frames, first after 6 frames (200 ms)
```

## Three defects found while building it, each by measurement

### 1. The search did not magnify

Sizing the crop to a fixed output resolution gave a 785×693 region a **scale of
1.00** — no magnification at all. A subject who left at 480 px and returned at
140 px was never found, because the search was handing the detector the same
pixels that had already failed.

The scale must come from the **subject's expected height**, not the region's
size. Fixed by targeting 500 px of subject and searching several assumed sizes,
since a subject invisible to the full-frame detector is usually *smaller* than
when last seen.

### 2. A fixed give-up stranded the user

Searching stopped after 3 seconds of absence, to protect the frame rate.
Measured, returning to the same spot:

| away | returns at 240 px | returns at 480 px |
|---|---|---|
| 1 s | recovered | recovered |
| 5 s | **STRANDED** | recovered |
| 15 s | **STRANDED** | recovered |

The 480 px cases only recovered because the full-frame detector can see a
subject that large unaided — the search had already stopped. Anyone returning
beyond ~2.5 m was stuck exactly as before the fix.

Searching now never stops. The last known position stays valid indefinitely: the
subject left a room whose camera did not move.

### 3. The back-off broke recovery

The interval doubles after each failed search. Doubling it *during* the eager
first second meant that by the time the subject returned, the interval had
already reached maximum — so nothing looked for them. **The cost control
reintroduced the exact stranding the module exists to prevent.**

## Cost, swept rather than guessed

Subject away 20 s, then returns at 240 px and stays:

| max interval | empty room, per frame | found after |
|---|---|---|
| every frame | 43.7 ms | — |
| 0.25 s | 12.06 ms | 0.13 s |
| 0.50 s | 7.93 ms | 0.07 s |
| **1.00 s** | **6.36 ms** | **0.13 s** |
| 2.00 s | 5.81 ms | 0.93 s |
| 4.00 s | 4.34 ms | 0.83 s |

1.0 s is the knee — cost has nearly bottomed out and recovery is still within a
tenth of a second. Past it the cost barely improves while the delay grows
sevenfold.

**An earlier sweep appeared to show longer intervals stranding the user. That
was a defective test** — it declared failure after one second of standing still,
which is shorter than the interval being measured. The interval is a worst-case
delay, not a failure.

The search costs nothing while tracking is working, because it only runs on
frames where the subject was not found.

## What this did and did not fix

**Walking towards and away: was never broken.** 640 px out to 100 px and back,
112/112 frames tracked, with or without the search. Once locked on, tracking
holds to 60 px — distance only matters if the lock breaks.

**Fast motion across the frame: improved.** MediaPipe derives the next crop from
the previous frame's landmarks expanded 1.25× (`RectTransformationCalculator`,
`scale_x/y: 1.25`, `square_long: true`), with **no motion model at all** — it
centres on where the subject *was*. The model was trained with only ~10%
shift/scale augmentation.

| motion per frame | before | after |
|---|---|---|
| 0.03 body widths | 44/44 | 44/44 |
| 0.17 body widths | 6/7 | **7/7** |
| 0.28 body widths | 3/4 | **4/4** |
| 0.42 body widths | 1/2 | **2/2** |
| 0.62 body widths | 1/2 | **2/2** |

Tracking starts dropping frames at **0.17 body widths per frame**, which is the
1.25× margin behaving exactly as the source predicts. The search recovers the
dropped frames, and the locator's velocity extrapolation aims the search where
the subject is *going* — the piece MediaPipe's own design lacks.

## Alternatives researched and not adopted

- **RTMPose-s** — 72.2 AP COCO at 13.89 ms on a Snapdragon 865, and its
  reference pipeline re-runs its detector **every 5 frames** rather than
  trusting tracking indefinitely. The strongest deployable alternative, and the
  source of the periodic-re-detection idea used here. Not adopted now: it is a
  whole second model stack, and the measured failure is fixed without it.
- **RTMO** — 74.8 AP, 141 FPS, but on a V100. Its advantage is multi-person,
  which does not apply.
- **ViTPose++ / Sapiens** — ~80 AP and better, but server-grade. Sapiens is
  genuinely useful as an *offline* labeller at 1K resolution.
- **YOLO-pose** — 57.2 AP for the nano variant, against RTMPose-s's 72.2. It is
  bottom-up, so a distant subject gets a small share of the feature map. Its
  real role would be as a person *detector*, not a pose model.
- **SmoothNet** — 77–95% jitter reduction, but non-commercial licence.

## Not established

- All of this uses a real photograph scaled synthetically, not a person
  genuinely standing at 2 m and 5 m. Scaling reproduces the size change but not
  perspective, focus falloff, or motion blur.
- The metre figures are converted from pixel heights with a rough field-of-view
  assumption. The pixel heights are what were measured.
- **Rep counting on a real moving human remains unvalidated**, as before.
- Fast-motion numbers come from translating a rigid image, which has no motion
  blur. Real fast movement is blurred, and blur is absent from BlazePose's
  published augmentation list — a known gap, not measured here.
- Camera exposure was not investigated. Research suggests pinning a minimum
  frame rate to cap exposure time is the highest-value change for blur, and it
  is untested here.
