# Handover

Read this first. Everything below was measured on this machine.

**Try it:**

```
cd engine_body_sim
python pose_sim.py --exercise squat
```

Squat. It counts, says "not deep enough" on a partial, comments only when a
repetition differs from your others, and pauses with a reason if tracking goes
wrong. `--calibrate --profile me.json` if you want thresholds fitted to you —
though the velocity counter no longer needs it. `--debug` shows the diagnostic
overlay, which is off by default.

**Status: the user has run this and confirmed it works well.** The body engine
is no longer the active work. The active work is the first game — see
"THE ACTIVE TASK" immediately below.

---

## THE ACTIVE TASK — the first game

**Nothing has been built yet. This is a specification agreed with the user, not
a description of existing code.**

### Where it goes

**`C:\dev\games\reach\`** — under `C:\dev\`, **outside** `C:\dev\psybot\`, at
the user's explicit request.

The shape the user asked for: one **games folder**, with **each game in its own
subfolder** inside it. There will be several games for different states —
tension, inattention, low mood — and the whole point of the separation is that
one broken idea cannot take the others down. `reach` is the first; the folder
must be created so a second and third can sit beside it without rearranging
anything.

**Python first**, exactly as `engine_body_sim` was. The user can play it within
the hour and say "this is boring" or "this is too hard" before anything is
invested in Dart. That sequence is the one that already worked on this project.
Port to phone only once it is good.

How it reaches the engine: import from `C:\dev\psybot\engine_body_sim\`. The
game must **not** copy engine files into itself — a copy diverges silently, and
every fix measured over three sessions would stop reaching it.

### Why this game, and why not the alternative

The user's own two candidates were "copy a pose" and "touch a square". **Touch
a square is the stronger of the two**, for a reason specific to this
application:

Pose imitation requires comparing a body against a reference pose, which means
judging *correct* against *incorrect* on something that legitimately varies
between bodies — flexibility, proportions, injuries. In a mental-health app that
turns into the bot passing judgement on the user's body.

Touching a square judges nothing. **Touched or did not touch.** Binary, always
explicable, and there is no way to "do it wrong".

### The rules, as decided by the user

| Question | Decision |
|---|---|
| Which body part | **Any part counts.** Hands are the most likely, but a head touch, knee touch or foot touch all score |
| Mirrored | **Yes.** The player sees themselves and reaches; without mirroring they move the wrong way |
| Target size | **Scales with distance** — the further the player is from the camera, the smaller the squares, to keep the challenge comparable |
| Device | Built for a **phone, propped up**. Tested on the laptop for now |
| Registering a touch | **Instant.** No dwell time. A lucky touch while waving counts — the user explicitly wants this, for fun |
| Timed mode | On timeout the square **changes colour, disappears, then the score is shown** |
| Untimed mode | For relaxation. Ends only when the player **raises the stop gesture** |
| Target placement | **Fully random** — near, far, and very far, for variety — but **never somewhere that needs impossible effort**, e.g. two body-heights away |

### Finger commands during play

The finger numbering is already built and measured in `hands.py`:
**little finger = 1, ring = 2, middle = 3, index = 4, thumb = 5.**

| Fingers raised | Action |
|---|---|
| 3 + 4 + 5 | **Pause** the game temporarily |
| 4 + 5 | **End** the game |
| 2 + 3 + 4 | **Resume** after a pause |

Note for whoever implements this: `hands.py` currently defines the stop gesture
as 1 + 5 held for 1.5 s. These three new combinations are **different from
that** and are game-specific. The hand reader samples every 4th frame
(measured: 11.2 ms inference, so ~1.5 ms/frame amortised), and a gesture must be
*held* to avoid firing on a hand passing through a shape.

### Two modes, one engine

Agreed in discussion: the same mechanic runs in **race mode** (timer, score) and
**calm mode** (no timer, the square waits, the point is simply to arrive). The
programming difference is only the timer, so the cost is near zero — but the
psychological difference is large. A timed, scored game is good for inattention
and low energy, and may be actively **harmful for anxiety**, where time pressure
adds to what the user already feels. The bot picks the mode from the state it is
responding to.

### The design decision that matters most technically

**Place targets in body units, not screen units.**

If a square appears at a fixed screen position and the player is standing far
back, it may be **physically out of reach** — they fail through no fault of their
own, which is the worst feeling a game can produce. If instead the position is
computed from their own body ("one arm-length up and right of the shoulder"),
it is reachable at any height and any distance from the camera.

The engine already stores everything in torso units for exactly this reason, so
the machinery exists.

### What is known about the parts this game depends on

From measurements already in this document:

- **Wrists are the strongest extremity we track.** Hands are the safe default.
- **Ankles are documented as the first thing to fail** — small, often occluded
  by furniture, and frequently outside a phone's framing. Foot targets should be
  measured before being relied on, not assumed.
- **Landmarks that leave the frame are now discarded**, so a target near the
  edge will simply not register a touch from a limb that is out of shot. This is
  correct, but the game must not place targets where that happens constantly.
- **A held (occluded) landmark must not score a touch.** `OcclusionReport`
  already distinguishes seen / held / discarded; the game should require `seen`.

### Open questions the implementer must still resolve

These were **not** settled in discussion and must not be guessed:

1. **What is the reachable envelope, measured?** A standing person cannot reach
   most of the frame. Target placement needs the real envelope, and the existing
   clips are people exercising, not people reaching — so they understate it.
   This needs a short recording of someone actually reaching.
2. **How small can a target get before touching it is luck?** Ties to the
   distance-scaling rule.
3. **Does an instant touch produce false positives at the measured 1.09 px
   noise floor?** Probably not at any sane target size, but it is unmeasured.
4. **What happens on failure?** The user said the square changes colour and
   disappears. Whether anything else is shown — and the strong recommendation
   from the discussion that there should be **no "you lost" at all** — is open.

### The rule that governs all of it

`AGENTS.md` forbids inferring a mental state from movement. **The game is not a
measurement and not an assessment.** It is something to do. The bot must not
comment on how the user's body performed, and offering the game must be a
suggestion that can be declined with one tap and no insistence.

---

## The two findings that reframed everything

### 1. MediaPipe tracks real photographs at 0.997. It tracked our rendered test figures at −0.04.

Every accuracy number measured before this was measured on rendered figures, and
was therefore measuring nothing. **The Round 2 knee-accuracy figure of 2.6° is
withdrawn and must not appear in the report.**

The tracking pipeline was never the problem. The test imagery was.

### 2. Every confidence gate in the system was reacting 233 ms late.

MediaPipe smooths its visibility scores internally with `alpha: 0.1` — read from
the shipped graph, not assumed. That EMA takes **7 frames to fall halfway** and
22 to fall 90%. So for roughly seven frames the engine was acting on landmarks
the model already knew were wrong — long enough to count a repetition that never
happened.

Bone-length constancy and velocity outlier checks compare consecutive frames and
react in **one**.

---

## The change that most affects real users

**Repetition counting no longer needs calibration.**

A position threshold asks "is the knee below 100°?", which requires knowing what
that means for this person. A velocity reversal asks "did they turn around" —
the same question for every body.

Measured on identical movement, neither counter calibrated:

| User bottoms at | Threshold counter | Velocity counter |
|---|---|---|
| 80° | 5/5 | 5/5 |
| 91° | 5/5 | 5/5 |
| **107°** | **0/5** | **5/5** |
| **122°** | **0/5** | **5/5** |
| **138°** | **0/5** | **5/5** |

Three user types counted **nothing at all** before. The app silently told them
they had done nothing.

Both counters still run. The velocity one counts; the threshold one knows *how
deep* and can say "not deep enough". Zero false positives across standing still
at four noise levels, walking in place, sitting down, and arm movement.

---

## Measured performance

### Where frame time goes (`lite` model)

| Stage | Time | Share |
|---|---|---|
| **Inference** | **13.81 ms** | **78.1%** |
| Drawing | 1.76 ms | 9.9% |
| Conversion | 1.35 ms | 7.6% |
| Session logic | 0.62 ms | 3.5% |
| Angles | 0.16 ms | 0.9% |
| **Total** | **17.70 ms** | **56.5 fps** |

**All of our own logic is 4.4% of a frame.** Optimising it would be pointless;
inference is the only lever that matters.

### Model choice — measured, and the result was a surprise

| Model | Median | Jitter | Knee σ | vs heavy (knee) |
|---|---|---|---|---|
| **lite** | **25.2 ms** | 0.70 px | **0.43°** | **0.2°** |
| full | 30.6 ms | 0.72 px | 0.79° | 1.3° |
| heavy | 70.7 ms | 0.46 px | 0.22° | — |

**`lite` is 18% faster than `full` and *better* on knee-angle stability** — the
quantity every threshold keys on. `full` cost 5.4 ms per frame for nothing. Now
the default.

`heavy` is genuinely more stable but 2.8× the latency, against a noise budget
that already has 9× margin.

### Noise floor on a real human

| | |
|---|---|
| Per-landmark jitter (untracked) | **1.09 px** — budget is 10 px |
| Knee angle σ | **0.66°** — published LoA is ±12° |

**9× margin.** The design is not near its limit.

### Capture backend and frame size (`docs/measurements-capture-display.md`)

The preview opened bigger than the screen. Two documented assumptions were
wrong, and fixing them doubled the capture rate.

| | DSHOW | MSMF |
|---|---|---|
| Frame rate | 15.2 fps | **29.1 fps** |
| Open latency | 274–319 ms | **52–134 ms** |
| 8-index scan | 1651 ms | **1472 ms** |

The code preferred DSHOW on a comment claiming MSMF was slow to open. **MSMF is
faster on every axis here**, including that one. Order is now MSMF → DSHOW →
ANY; DSHOW stays as a fallback because the claim may hold on other cameras.

**The camera ignores every resolution it is asked for** — 640×480, 1280×720 and
1920×1080 all return 2560×1440, in YUY2 and MJPG, on both backends. The
`CAP_PROP_FRAME_WIDTH` call was dead code and is gone; frames are resized after
capture.

Detection height, knee angle vs detection at full 1440 over 180 real frames:

| Height | mean abs diff | p95 |
|---|---|---|
| 1080 | 1.12° | 3.40° |
| **720** (default) | **1.19°** | 4.29° |
| 360 | 1.22° | 4.16° |

**The difference does not grow as the frame shrinks**, so it is resampling
noise, not lost resolution. Detection confirmed deterministic first (same input
twice agrees to 0.0000°).

**Shrinking does not speed up inference** — ~12 ms at every size, because
MediaPipe rescales to its own fixed input internally. End to end 29.6 → 31.0
fps. The reason to downscale is that the window fits the screen; claiming a
speed win would be false. Full loop against the real camera: **35 fps steady
state**.

### The jump glitch (`docs/measurements-jump-guard.md`)

`SubjectSwitchDetector` fired mid-jump, resetting the counter and discarding the
repetition. `hip_rate` was 9.0, documented as leaving "generous headroom for a
jump". Measured, an explosive push-off reaches **13.92 /s** — the threshold sat
*inside* the population it was meant to be above.

Raising it is not sufficient: an explosive jump (13.92 /s) and a switch to a
same-build person half a torso away (15.00 /s) **overlap**. Direction and motion
prediction were both tried and both failed on measurement.

**A persistence rule was tried and rejected by test.** Requiring two consecutive
anomalous frames sounds right — a jump's peak is transient, a switch persists —
but a genuine switch is anomalous on exactly *one* frame, so the guard stopped
detecting switches entirely. The reasoning confused the person persisting with
the rate staying high.

Measured over **399 physiologically-timed movements** with landmark noise at the
1.09 px measured on a real human:

| | torso | hip | false positives | switches caught |
|---|---|---|---|---|
| before | 7.0 | 9.0 | **186 / 399** | 31 / 35 |
| **after** | **4.0** | **20.0** | **0 / 399** | 26 / 35 |

186 false positives removed, 5 detections lost. Ported to Dart identically.

**Cannot be fixed:** a switch to a same-build person within half a torso is
indistinguishable from a jump on these quantities. Stated, not tuned away —
tuning it away is what caused the glitch.

The reliability and bone-length monitors were checked on the same sequences and
do not fire (89/89 and 84/84 frames usable).

### Stability, occlusion, fingers (`docs/measurements-stability-hands.md`)

**Nothing enforces a skeleton.** Verified from MediaPipe's own smoothing proto:
the only options are `NoFilter`, `VelocityFilter`, `OneEuroFilter`, all
per-landmark-**independent**. No bone-length term at all. `pose_world_landmarks`
is an output space, not a world model. Measured consequence: bones vary 0.72% on
a *still* subject and the forearm swings **179%** during movement.

Fix: learn bone lengths over ~2 s (median, torso-relative, L/R averaged), then
keep each bone's direction and substitute its calibrated length.

| | bone variation raw | constrained | jitter |
|---|---|---|---|
| still | 0.72% | **0.20%** | 0.353 → **0.246 px** |
| curl | 12.59% | **3.45%** | |

**Angles change by 0.000°** — they are functions of directions alone. Getting
there found a defect: measuring direction from the *already-moved* parent mixed
two coordinate frames and changed the elbow angle by up to **175°**.

Smoothing the torso scale was **tried and rejected** — on a real subject it is
already stable at 0.20% and every filter made it worse.

**Occlusion: the model's confidence is sometimes a lie.** Covering an arm moved
the wrist **149 px (a full torso)** while reporting **0.97** confidence — our 0.5
gate accepted it. (Legs behaved correctly, which is worse: it cannot be relied on
either way.) Now checked independently by bone length and teleport rate, then
**held** at the last trusted position, confidence decaying to unusable at 0.8 s
and abandoned at 1.5 s. Held limbs draw dashed; measurements abstain.

**Fingers: ML Kit has no hand API** — confirmed from the Maven group index, no
`hand-*` artifact exists. Pose gives 3 points per hand, no joints.
`tasks-vision` 1.0.0 gives 21. **Decided: migrate Android to MediaPipe** — not
done, needs a phone.

Full-frame hand detection **fails below a 300 px hand**; a person at exercise
distance has one 40–70 px across. The wrist crop works down to 40 px, so it is
**mandatory, not an optimisation**. Sampled every 4th frame (11.2 → 1.5 ms
amortised; 13 samples per 1.5 s hold). Stop gesture = fingers 1+5, held 1.5 s,
quits. End to end: **36.5 fps**.

### Tracking range (`docs/measurements-tracking-range.md`)

**The detector and the tracker have a 6.7x range gap.** The detector is
face-proxy based (Google's paper: *"a fast on-device face detector as a proxy
for a person detector"*) and runs on the whole frame; the landmark model only
ever sees a crop rescaled to 256x256.

| | works down to | roughly |
|---|---|---|
| fresh detection | **400 px tall** | 2.5 m |
| established tracking | **60 px tall** | 17 m |

And the detector is gated off entirely while tracking succeeds, with **no
periodic re-detection anywhere** in MediaPipe's graph. So:

```
tracked at 240 px (~4 m)     ok
steps out for 1 s            lock lost
returns at 240 px            CANNOT RE-ACQUIRE
walks back to 480 px         recovered
```

Fixed by searching an upscaled crop of the last known region instead of the
whole frame: **0/20 frames recovered before, 14/20 after**, first within 200 ms.
Works down to a 60 px subject.

Three defects found by measurement while building it: the search initially
**did not magnify at all** (scale 1.00 — handing the detector the pixels that
already failed); a 3-second give-up **stranded** anyone returning beyond 2.5 m;
and the cost back-off ran during the eager window, reintroducing that same
stranding. Interval swept, 1.0 s chosen — 6.36 ms/frame over an empty room,
recovery within 0.13 s.

**Walking in and out was never broken** — 112/112 frames. Fast motion drops
frames at **0.17 body widths/frame**, which is MediaPipe's 1.25x crop margin
behaving as its source predicts; the search now recovers those.

Researched and not adopted: RTMPose-s (72.2 AP, 13.89 ms on a Snapdragon 865 —
the source of the periodic-re-detection idea), RTMO, ViTPose++, Sapiens,
YOLO-pose, SmoothNet.

### The skeleton is drawn as a stick figure

The trunk was four bones forming a quadrilateral, with a joint dot at each
corner of the chest. There is no shoulder-to-hip bone in a skeleton; the ribcage
is not a hinge. It is now a spine from mid-hip to mid-shoulder — the axis every
measurement already uses — plus two cross-bars and a neck stub. Verified on
rendered pixels: spine and bars 100% inked, both sides of the old box clear at
their midpoints.

---

## What was built

| Module | What it does |
|---|---|
| `exercise_session` | **The single entry point.** One call per frame |
| `velocity_reps` | Calibration-free rep counting from direction reversal |
| `reliability` | Bone-length + velocity gates that react in 1 frame |
| `hold_detector` | Timed holds for stretching, plank, balance |
| `form_score` | DTW comparison of each rep against the user's own first |
| `angles` | Joint angles carrying their weakest landmark's confidence |
| `calibration` | Per-user thresholds, refusing loudly when unusable |
| `movement` | Hysteresis counting, jumps, centre of mass, guards |

| `body_guards` | Subject switch, viewpoint gate, median filter, centre of mass |
| `jump_detector` | Jump detection with a free-fall physics check |
| `user_profile` | Persisted thresholds — angles only, nothing identifying |

**Dart is now feature-equivalent to the Python simulator: all sixteen modules
present, zero gaps.** That was not true earlier in the session — nine modules
existed only in Python, so the app that ships had no subject-switch detection,
no viewpoint gate, no centre of mass and no jump detection at all.

**570 Python checks, 181 Dart tests, analyzer clean.**

### Seven exercises

`squat`, `pushup`, `armraise`, `situp` are counted; `plank`, `stretch`,
`balance` are timed. Each names a **large joint in the camera plane**, where 2D
pose estimation is most accurate. Movements needing rotation, small joints or
depth are deliberately absent — a single phone camera cannot measure them
honestly, and a test enforces that.

### Hold detection measures displacement, not speed

Frame-to-frame speed measures the *camera's noise*, not the person's stillness:
0.24 torso lengths per second at 1 px of landmark noise, rising to 2.44 at
10 px. Any fixed speed threshold therefore encodes an assumption about camera
quality — at 6 px a motionless statue registers as moving.

Displacement from an anchor pose does not degrade this way, because zero-mean
noise cancels instead of accumulating. Verified correct from 1 px to 15 px.

### Form scoring is free because of where it runs

It runs *after* a repetition completes, on a finished trajectory — 0.35 ms once
per rep against 13.81 ms of inference every frame. That is what makes DTW
usable: it is O(N×M) and non-causal, which disqualifies it for live detection
and makes it exactly right here.

The reference is the user's **own first complete repetition**, never a
population norm — so the comparison cannot judge body proportions, mobility or
camera placement, none of which the engine can see.

---

## The most dangerous defect found

**A single NaN coordinate silently stopped the repetition counter.**

An EMA poisoned with NaN stays NaN for ever — `NaN + α(x − NaN)` is NaN. Every
later sign comparison is then false, so no movement direction is established and
counting just stops. **Nothing is logged. Nothing looks wrong.**

Measured: one NaN landmark mid-session took counting from 7 to **4** of 7, and
the three lost repetitions were every one that followed.

Infinity is equally damaging and less obvious, since `inf − inf` is NaN.

Fixed with one shared guard at the public entry points in both languages. Now 6
of 7 — the repetition containing the fault is correctly refused by the
reliability gates, and the ones after are unaffected.

Non-finite values arrive from real sources: a tracker that cannot place a
landmark, or a landmark sitting exactly at a joint centre.

## Defects found and fixed this session

Eleven. All caught by pushing the code, none by reading it.

1. **The viewpoint gate refused users facing the camera** — reported 64° of yaw
   when the truth was 6.7°. The nose follows the *head*, so glancing aside read
   as turning away. Now uses 3D torso yaw.
2. **Phantom complaints on perfect reps** — the dwell counter incremented every
   held-back frame, but waiting out the dwell is what a *normal* rep does. Five
   ideal squats produced four false complaints.
3. **The free-fall check barely worked** — a 20 cm tiptoe rise read as a jump in
   **8 of 8 runs**. The assumption that tiptoes fits near-zero acceleration was
   wrong; measured, it fits 0.68.
4. **Subject-switch fired on every squat** — folding the body changes apparent
   torso length by over 100%; what distinguishes movement from substitution is
   *rate*, not size.
5. **Standing still scored one phantom rep, every time** — the first reversal
   had no reference to measure travel from, so the check was skipped.
6. **Every rep reported "slower than your first"** on a constant-tempo set — the
   first captured record is truncated (24 frames vs 36) because counting begins
   mid-movement.
7. **A crash waiting to happen** — the rep banner referenced a colour constant
   that does not exist. It would have thrown on the first successful rep.
8. **The bottom could be touched rather than held** — a 20 ms bottom counted.
9. **Silence on impossible movement** — bouncing produced no feedback at all.
10. **Two wrong definitions of what closes a repetition** — gave 8 and then 4
    where the answer is 5.
11. **The test fixture was not a body** — it dropped the hips while holding the
    knees fixed, so the torso lengthened 100% and the femur shrank to a ninth.
    The bone-length monitor flagged 89 frames of it and was entirely right.

Plus five wrong test fixtures, each corrected with the reason recorded.

---

## Techniques evaluated and rejected

Rejected with reasons, so they are not revisited blindly:

| Technique | Why not |
|---|---|
| Butterworth filtering | The "6 Hz standard" is **non-causal**. The causal version costs 37–70 ms — 4–18× current lag |
| Savitzky-Golay (live) | 130–170 ms lag. Fine for offline analysis only |
| Kalman (CV or CA) | Lags *most* at the rep turnaround — exactly where precision is needed |
| SmoothNet | ~1% gain on a good backbone, 133–533 ms lag, no mobile runtime, non-commercial licence |
| RepNet | Outputs a *period*, not a rep *event*. ResNet-50 on pixels |
| Heatmap entropy | BlazePose **discards the heatmap head at inference** — the tensor does not exist |
| TTA / ensembles | 2–N× inference for a confidence number |
| RTMPose / MoveNet | Faster, but **17 keypoints against MediaPipe's 33** — we would lose the feet, hands and face that framing, centre of mass and extremity boxes depend on |
| YOLO-pose | 640×640 input, ~20× the FLOPs, and AGPL |
| **3D world landmarks for position invariance** | Tried against the user's reported bug. **Worse**: 27.3% bone spread against 5.4% for pixels on a distance change, and no better on position. The name implies position independence; measured, it is not |
| **Shoulder-width scale fallback** | For waist-up framing where the hips are outside the picture. Made `turning_1` **four times worse** (44.1% → 164.9%) by mixing two scales within one clip |
| **Faster calibration refresh** | 19.5% bone spread at 90 frames against 3.5% at 300. A refresh discards a settled median and rebuilds it from whatever the subject happens to be doing, so frequent refreshes keep the model permanently half-learned |
| **Running-mean "leash" on the anchor** | Built to stop drift, measured to change it **not at all** (4.37 px with and without), because the drift is chance threshold crossings rather than a random walk. Cost 0.75 px of lag and broke release monotonicity |
| **Per-frame upscaled crop for accuracy** | +0.015° — no measurable gain |

---

## Session: off-frame landmarks, anchoring, and the 28-clip survey

### The user's report: "random marks when I jump or leave the edges"

**Cause, measured.** MediaPipe places landmarks that have left the picture at
coordinates **outside the image**, and still reports high confidence:

| landmark | x | past the edge | confidence |
|---|---|---|---|
| leftElbow | 1286 | 6 px | 0.98 |
| leftWrist | 1341 | 61 px | 0.94 |
| leftFootIndex | 1355 | 75 px | 0.54 |

Nine landmarks outside a 1280×720 frame. The drawing gate was 0.2, so every one
was drawn.

### `presence` does not solve this, and it looked like it would

MediaPipe documents two separate scores: `visibility` means "occluded by
something", `presence` means "present on the scene". The second reads exactly
like an out-of-frame signal, and the Python API does expose it.

Measured, for landmarks genuinely outside the image: **presence 0.73–0.99.**

**There is no model output that distinguishes "outside the picture" from
"hidden behind something".** The only reliable test is geometric. Recorded
because the field name strongly implies otherwise and a future reader will
reasonably try it again.

### The rules now applied, in order

| condition | action |
|---|---|
| coordinate outside the image | **discard**, and erase that landmark's history |
| hidden, observed 20+ times | **hold** at the last trusted position, decaying |
| hidden, observed fewer | **discard**, invent nothing |

Leaving the frame erases the memory because a limb that walks out of shot is
*gone*, not occluded — holding its last position would pin a hand to the edge,
and on return that position is stale by however long they were away.

20 frames rather than one because the model reports **0.97 for a wrist a full
torso length out of place**, so a single confident frame proves nothing.

### Defects found by measurement while building this

1. **The edge margin was reasoned, not measured.** 2% of a 1280-wide frame is
   26 px against a **1.09 px** noise floor — 24× too generous — and it
   measurably admitted two genuinely off-frame landmarks. Now 0.2% (2.6 px).
2. **Three drawing steps ignored the discard flag entirely**: hand/foot boxes,
   angle labels and corner brackets drew **4067 pixels** with every landmark
   discarded. Now zero. The angle labels mattered most — printing a number next
   to a joint asserts that a measurement was made.
3. **Bones were drawn to unclipped endpoints**, skewing the limb towards
   whichever invented coordinate it landed on. That is what put the skeleton
   beside the body rather than along it. Now clipped (Liang-Barsky).

### Joint anchoring — and why a deadband was rejected

The margin between noise and real movement is thin:

| joint | still, p95 | moving, median | ratio |
|---|---|---|---|
| **leftKnee** | 0.0074 | 0.0119 | **1.6×** |
| leftWrist | 0.0072 | 0.0529 | 7.4× |

The left knee has only a 1.6× margin, and the knee is what every threshold in
this engine reads. A deadband sized to remove the shiver would eat part of a
slow squat.

What measurement chose instead — hold while quiet, release *completely* once
movement clears the noise floor, blend between:

| design | still jitter | cut | knee error |
|---|---|---|---|
| none | 0.00191 | — | — |
| deadband 0.0074 | 0.00061 | 68% | 0.080° |
| **adaptive 0.0050/0.015** | **0.00032** | **83%** | **0.146°** |
| adaptive 0.0074/0.020 | 0.00018 | 91% | 0.285° |

0.146° against a knee noise floor of **0.66°** — about a fifth of the noise it
competes with.

### A throughput scare that was a measurement error

The synthetic harness reported 11 fps against 31.2 earlier — an apparent severe
regression. Profiled, the new code costs **under 1 ms/frame** in total.
Stashing every change and re-running on the committed baseline gave **10.0
fps** — slower than the new code. The 31.2 figure had been taken under
different machine load and was never a valid comparison. Steady-state
per-frame timing is the reliable measure: **25.3 fps**.

---

## Session: 28 real clips

Everything before this was measured on a single photograph, transformed. 28
free-licence clips were downloaded — people walking toward and away, hiding an
arm behind the back, turning, stretching, dancing, squatting — and run through
the real pipeline. **All 28 were deleted afterwards at the user's request.**

### The user's report: "arm length is wrong when I change position"

Confirmed. Same person, same size, seven positions in one frame:

| position | leftElbow | leftKnee |
|---|---|---|
| centre | 0.5705 | 0.7342 |
| high | 0.6158 | 0.7671 |
| low | 0.5715 | 0.6654 |

**14.2% spread from position alone**, against 3.45% honest movement variation.
The cause is perspective: a person at the edge of frame is viewed obliquely, so
their limbs really are foreshortened.

### Four defects found

**1. Calibration gates were far too strict.** The torso gate rejected a whole
frame if any one of four landmarks fell below 0.5:

| clip | frames rejected |
|---|---|
| dance_1 | 196 of 200 (98%) |
| stretching_2 | 195 of 200 (98%) |
| walk_toward_3 | 0 of 200 |

The body model never built, so bones ran unconstrained at 40–55%. Now 0.2 plus
a geometric in-frame check.

The per-bone gate was 0.5 for the same bad reason. **Lower is better on both
counts**, because the statistic is a median and a median wants more samples,
not purer ones:

| gate | clips built | bone spread |
|---|---|---|
| **0.2** | **6 of 8** | **6.3%** |
| 0.5 | 5 of 8 | 11.0% |
| 0.6 | 5 of 8 | 20.1% |

**2. A held limb was stored in pixels.** On a clip where the torso spanned 33
to 182 px, a held elbow gave a bone-ratio spread of **553.9%** against 29.2%
where the same bone was seen. The limb had not moved; the reference had
doubled. Holds now store an offset from mid-hip in torso units, rebuilt each
frame. That introduced 16 off-frame escapes of its own, so the rebased position
is re-checked.

**3. The anchor pushed landmarks out of frame** — 10 landmarks 0.9 to 2.2 px
past an edge, having already been passed as drawable. Clamped when the
detection was inside; deliberately **not** when it was genuinely outside, which
would disguise it as visible.

**4. No guard looked at joint angles.** 199 changes over 40° between
consecutive frames, worst **153.8°**. Both existing guards missed every one,
structurally: they test each landmark in isolation, and a knee swinging 153°
moves only half a torso.

| percentile | deg/frame |
|---|---|
| p50 | 0.7 |
| p90 | 7.0 |
| p99 | 51.5 |
| max | 163.3 |

A sevenfold gap between p90 and p99 is two populations. `MAX_ANGLE_RATE =
500 deg/s`. The guard **rejects rather than corrects**, and a rejected value is
not recorded as the reference, so one bad frame cannot drag the comparison.

### Results

| metric | before | after |
|---|---|---|
| off-frame drawn | 10 | **0** |
| angle jumps >40° | 199 | **9** |
| yoga clip | 17.8% | **2.4%** |
| exercise_3 | 29.5% | **4.6%** |
| turning_1 | 322.0% | **160.6%** |

### Hidden-arm behaviour, traced

On the hand-behind-back clips the state sequence is
`seen → held → discarded → seen`, and **`discarded → held` never occurs**: a
limb is never revived from nothing.

### Adaptation speeds, swept

| setting | was | now | evidence |
|---|---|---|---|
| calibration learn window | 60 | **30** | 2.4% vs 3.5%, ready twice as fast |
| calibration refresh | 300 | 300 | 90 frames measured much worse |
| anchor quiet/release | 0.005/0.015 | unchanged | no measurable difference across 6 settings |
| trust threshold | 20 | unchanged | no measurable difference across 5 settings |

A longer learn window is **not** a better median — it spans more of the
subject's movement, so more foreshortened samples drag it.

### Limitations, stated not hidden

- **Waist-up clips cannot be constrained.** On `stretching_2` the hips sit 34%
  below the bottom edge in every frame. No torso, no scale, no model. A framing
  requirement, not a bug.
- **The worst remaining clip (497%) is out of range** — 96% of its frames have
  the body under 400 px, below the measured detection floor, as small as 74 px.
- Both remaining high-spread clips are one of those two cases.

---

## Session: the Dart port

**The finding that prompted it:** three sessions of work existed only in the
Python simulator. Dart — which is what ships — had **none** of it. No off-frame
rule, no occlusion tracker, no joint anchor, no angle-rate guard.

Now ported: `occlusion.dart`, `anchor.dart`, `AngleRateGuard` in
`joint_angle.dart`, `isOutsideFrame` in `skeleton.dart`. All wired into
`ExerciseSession` in the measured order — skeleton model, then occlusion, then
anchor — with the angle guard between the angles and the side tracker. The
calibration gate changes came across too.

### Two real defects found while porting

1. **The guards keyed off `frame.timestamp`**, but several callers leave every
   frame on the same value — the test fixtures set all of them to epoch zero.
   Elapsed time drives the hold decay and the teleport limit, so the wrong
   clock **silently disabled both**. They now take the session clock, which is
   already passed in and is reliable.

2. **`ExerciseSession` never checked whether the joint it measures is in the
   picture.** With a subject shifted half a frame, the whole left side lands
   outside the image and is correctly discarded — and the session then reported
   `usable: true` with a null angle, which claims to be measuring while
   measuring nothing. It now refuses with a reason and keeps the counts.

### One existing test was wrong

`an unreliable frame is refused` shifted the body +0.5 to simulate a teleport.
In normalised coordinates that puts the **entire left side outside the frame**,
so it exercised the off-frame path, not the teleport guard it names. Measured:
+0.5 puts 6 of 13 landmarks outside, **+0.25 puts 0**. Corrected to +0.25.

### State

| | count |
|---|---|
| Python tests | **781** |
| Dart tests | **223** (was 196) |
| Dart analyzer | clean |

### Still not ported to Dart

- **`tracker.py`** — `SubjectLocator`, the re-acquisition search.
- **`hands.py`** — `HandReader` and finger gestures. **The first game needs
  this** for its pause/stop/resume commands.

---

## Overlay

The diagnostic overlay is now **off by default** and behind `--debug`: frame
rate, framing, landmark quality, joint angle, tracked side, viewpoint yaw and
the re-acquisition count are developer numbers and are clutter during a
workout.

What stays visible: rep counts, exercise state, any PAUSED reason, calibration
progress, hold timings, and the **REC indicator** — the user must always be able
to tell they are being recorded. `noSubject` became **"I cannot see you"**,
because a system that sees nothing otherwise looks identical to one that has
frozen.

Verified by running the real main loop and capturing every string that reached
`draw_panel`: **0 diagnostic lines leak** into the clean overlay.

---

## Audit: bugs looked for and NOT found

Worth recording so the same ground is not covered twice. Each of these was a
real hypothesis, measured, and disproved:

- **Stale state across a subject change.** Bone model: 0.0% error — the
  bone-length guard catches the mismatch. Occlusion trust: correctly rejected.
  Anchor: releases fully above threshold, so the jump passes through in one
  frame.
- **`discarded` not propagating.** It propagates correctly everywhere, because
  occlusion zeroes the confidence and every consumer already gates on that.
- **The hand reader receiving a discarded wrist.** Protected by the same
  zeroed confidence.

### Four hardenings applied anyway

None fire on current footage; all are one bad frame away from mattering.
`anchor.py` non-finite input reaching `_held`; `AngleRateGuard` dividing by a
zero `dt` when two frames share a timestamp (the recorder can produce this);
`movement.py` and `occlusion.py` using torso and mid-hip after only a partial
finite check.

### Known latent inconsistency, deliberately not fixed

`occlusion` uses `skeleton.torso_length` (geometric, always returns a number)
while `pose_sim` feeds `anchor` from `movement.torso_length` (which also gates
on confidence and returns `None`). `pose_sim` passes `or 0.0`, and the anchor
returns immediately when torso ≤ 0 — so on a frame where torso confidence dips,
**the anchor is silently disabled**.

Measured on 1334 frames: **0.0%** — it does not fire on well-framed footage.
Not fixed because changing the scale source changes anchoring behaviour and
there is no footage that shows the difference. Fixing it unmeasured would be a
guess.

### Technical debt worth knowing about

The drawing layer has **five undocumented magic gates** (0.15, 0.2, 0.25, 0.3)
alongside the defined `VISIBILITY_THRESHOLD`. Nobody knows why the head is at
0.2 and the fingers at 0.15.

---

## The one thing still unvalidated

**Repetition counting on a real moving human.**

Three attempts to fake it from a photograph all failed, and the reason is now
documented in `docs/validation-limits.md`: **a still photograph can validate
tracking, but not movement.** Rigid transforms preserve every joint angle by
construction. Changing an angle requires changing image content — new occlusion,
shading, silhouette — which cannot be faked from one frame.

The useful result from that failure: the reliability gates flagged **111 of 150
frames** of the corrupted attempt as untrustworthy. Without them, a confident
"5 of 5" from a body being torn apart would have been believed.

**The user has since run the simulator and confirmed it works well**, but has
not counted a known number of repetitions and compared. The claim stands
unvalidated. It matters less now than it did, because the first game does not
count repetitions — but it must not be quietly forgotten.

### Never measured at all

**Camera exposure and motion blur.** Absent from every measurement in this
project and absent from BlazePose's published augmentation list. Research
suggests pinning a minimum frame rate caps exposure time and reduces blur.
This is the **highest remaining accuracy gain that has never been tested**, and
it is cheaper than anything else on the list.

---

## Project shape

There is **no phone app yet**. `engine_body` is a pure-logic Dart package: it
takes a `BodyFrame` and produces counts and angles. Nothing yet produces a
`BodyFrame` on a phone — there is no camera layer and no native pose backend
wired up.

That is not a blocker for the current work. The plan the user described:

- a **chat bot** is the product
- it offers **games** when the user's state suggests one might help
- **more than one game**, for different states — tension, inattention, low mood
- each game lives in its **own folder**, so one failure cannot take the others
  down
- the body engine is the **sensing layer** those games use, not the product

The camera layer becomes urgent only when a game is good enough to put on a
phone. Building it now would be building an input for something that does not
exist yet.

---

## Next

1. **Build the first game** — `C:\dev\games\reach\`, Python, playable within the
   hour. Spec is at the top of this document. This is the active task.
2. **Measure the reachable envelope** before placing any target. The existing
   clips are people exercising, not reaching, so they understate it.
3. **Port `hands.py` to Dart** — needed for the game's finger commands, but only
   once the game proves worth porting.
4. **Camera exposure / motion blur** — the untested accuracy gain above.
5. **A real phone.** Never tested; `android-sdk` and `jdk-17` kept deliberately
   so an APK can be built immediately. Re-measure model choice on-device — the
   `lite` result should hold, but GPU delegate and NNAPI can change relative
   costs in ways a CPU cannot predict.
6. **Port `tracker.py` to Dart** — the last piece of the engine, and the least
   urgent until there is a camera feeding it.

### Decided but not implemented

**Migrate Android from ML Kit to MediaPipe `tasks-vision`.** The user chose
this. The reason: ML Kit has **no hand API at all** — verified against Google's
Maven index, there is no `hand-*` package in any version — and Pose gives only
three hand points with no joints. `tasks-vision` reached stable 1.0.0 in July
2026, gives 21 points per hand, and separates `visibility` from `presence`
where ML Kit returns a single value. **Needs a phone.**
