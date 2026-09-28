# Body tracking — decisions taken before any code was written

These are answers the user gave to the five forks in `AGENTS.md` section 8, plus
four follow-up questions that only became meaningful once the first five were
answered. They are recorded here so they are not re-litigated, and so that
anyone overturning one knows it was a deliberate choice and not a default.

Date: session 1. Nothing had been built at this point.

---

## The decision that changed the shape of the work

The section 8 questions assumed body tracking was **part of a game**. It is not.

The user's answer to "how much of the body" was:

> full body is recorded, and do not attach it to a game directly — there is a
> chance it will not be a game at all

and to "what does the game measure":

> the thing measures body movement, and we focus on that first; how we use it in
> games or whatever comes later is not important right now. The important thing
> is that it can detect the full body correctly. There may be more than one use.

So `engine_body` is a **sensing capability, not a game**. It detects the body and
publishes what it sees. Games, challenges, breathing exercises and anything else
are consumers built on top of it, later, in their own folders.

This is the right call for a project whose report may still change: a tracker
that works is useful under every version of the report. A game is only useful
under one.

**Consequence for the code:** `engine_body` contains no game logic, no scoring,
no target poses, no win condition. If a future folder needs those, it imports
`engine_body` and adds them itself.

---

## The nine decisions

| # | Question | Decision |
|---|---|---|
| 1 | Tracking stack | **ML Kit Pose Detection** (33 landmarks) |
| 2 | Body coverage | **Full body**, recorded; not bound to a game |
| 3 | What is measured | **Body movement itself**; uses come later |
| 4 | Where it runs | **Fully native**; Dart receives landmarks only |
| 5 | Shared camera window with face | **No** — body only, face measures before/after |
| 6 | Laptop simulation | **Useful, not mandatory** |
| A | Engine output | **Raw landmarks + derived signals** |
| B | Out-of-frame landmarks | **Framing gate + quality labelling** |
| C | Test platform | **Android only for now** |

---

### 1. ML Kit Pose Detection

Chosen over MediaPipe Pose and MoveNet/TFLite.

Reason: an official, maintained Flutter-facing path exists today. MediaPipe
would require hand-written Kotlin and Swift wrappers before a single landmark
appeared on screen. MoveNet gives only 17 landmarks — no hands, no feet — which
contradicts decision 2.

Costs accepted with this choice:

- Android and iOS only. **It cannot run on Windows.** This is why decision 6
  matters, and why the Dart core below is built to run without a camera.
- Adds roughly 5–10 MB to the app.

Not yet measured on a real device. `AGENTS.md` requires measurement before any
performance claim, so no frame-rate number is asserted here.

### 2. Full body, recorded, not bound to a game

The user wants the full skeleton captured, not the seated upper-body subset.

The honest trade-off, stated plainly: full body needs the phone propped up and
the user standing back roughly 2–3 m, with floor space. Many real sessions will
not have that. Decision B is the answer to what happens when they do not — the
engine reports the framing state instead of silently guessing.

### 3. Movement is the measurement

First goal is correct, stable detection of the whole body. Interpretation is
deliberately deferred.

### 4. Fully native, Dart receives points only

Camera frames never cross the platform bridge. Only 33 landmarks per frame do.

Converting YUV420 camera frames to RGB inside Dart is the single most common
cause of collapsed frame rates in Flutter camera apps. This design removes that
failure mode by construction rather than optimising it later.

### 5. Body only during the window

Face analysis runs in its own 60 s windows before and after; body runs during
the activity. Running both at once doubles cost and heat, and a moving user's
face is tilted and motion-blurred — poor face data anyway.

This also matches the `trend` reasoning in `AGENTS.md`: the before/after
difference is the valuable field, and it cancels the personal baseline.

### 6. Laptop simulation: useful, not mandatory

ML Kit will not run on Windows, so live skeleton on the laptop is not possible
under decision 1. What is possible, and what is built:

- The entire Dart core — normalisation, angles, framing gate, signals,
  smoothing, aggregation — is **pure Dart with no Flutter and no plugin
  dependency**. It runs in `dart test` on this laptop.
- Sessions recorded on a phone are saved as JSONL and can be replayed through
  the same core on the laptop, producing identical numbers.

So the laptop tests the logic; the phone tests the sensor. Nothing that can be
verified on the laptop needs a phone build to iterate on.

### A. Output: raw landmarks *and* derived signals

Raw landmarks are published for any consumer that wants geometry. Above them
sit derived signals, in the spirit of `affect.py` in the desktop engine.

All signals are computed in **torso units**, not pixels: the origin is the
mid-hip point and the scale is the shoulder-to-hip distance. This makes them
independent of how far the user stands from the camera and of the user's size —
the same property that makes joint angles robust.

**No signal here carries a psychological interpretation.** They describe motion
and geometry. `AGENTS.md` forbids quasi-clinical claims from a noisy sensor, and
that rule applies to the body exactly as it applies to the face. Any mapping
from movement to state must be measured before it is claimed.

### B. Framing gate and quality labelling

ML Kit always returns all 33 landmarks. When the legs are outside the image it
does not omit them — it guesses them, with a low `inFrameLikelihood`. A consumer
that ignores this will silently treat invented coordinates as measurements.

The engine therefore publishes, per frame:

- per-landmark likelihood and in-frame likelihood, unmodified;
- a **framing state**: `full` / `upperBodyOnly` / `partial` / `noSubject`;
- a **quality** value derived from the landmarks a consumer actually depends on.

Nothing is dropped and nothing is hidden. Consumers decide their own threshold.

### C. Android only for now

Kotlin layer written and tested. The Dart interface leaves room for iOS, and no
Android-specific type appears in the public API — but no Swift is written,
because it could not be built or tested from this machine and untested code in
a repository is a liability.

---

## What is deliberately not decided yet

- Frame rate and resolution: to be **measured** on a device, not guessed.
- ML Kit model: `PoseDetection` (fast) vs `AccuratePoseDetector`. Both will be
  measured on the same recorded session.
- Smoothing constants: the One-Euro filter parameters must be tuned against a
  recorded session with a known-still subject, so that sensor noise can be
  separated from real movement. Until that recording exists, defaults are
  marked as provisional in the code.
