# PsyBot — project handover

This file is read at the start of every session. It carries the decisions,
measurements and constraints that would otherwise have to be rediscovered.

Read it before doing anything. Where it states a measured number, that number
was obtained by running code, not by estimation — do not overturn one without
measuring again.

---

## 1. What is being built

An AI conversational agent for preliminary mental-health self-assessment and
evidence-based coping strategies, aimed at 18–35 year olds. It is a **final-year
degree project**, owned solely by the user.

The camera work in this repository is **not the product**. It is a sensing
layer that tells the chatbot how the user is responding, so the chatbot can
adapt its tone, pacing and choice of activity.

**Explicit boundary, from the project report:** the system is not a substitute
for therapy and does not produce a medical diagnosis. Every design decision
must respect that.

### Fixed constraints from the project report

| Item | Decision |
|---|---|
| Language | **English only** (Arabic/Malay excluded for scope) |
| Platform | **Mobile only** — no web, no desktop |
| Framework | **Flutter** (chosen over React Native for low-latency chat UI) |
| Model | Third-party LLM API, not a model trained from scratch |
| Methodology | Agile + prototyping |
| Deliverable | A working app **and** measurable results/reports |

**Desktop is for simulation only.** A Flutter desktop build may be used to
develop and test on the laptop, but the target is the phone.

---

## 2. The user's own words on scope

Written down verbatim because paraphrasing has already lost detail once:

- The phone version is *part of a larger chatbot app*; most of what it holds is
  information that helps the user understand their own state.
- The camera determines the user's state during use, then sends that to the
  chatbot so it can help the user and improve the quality of the interaction.
- Because of this, **most settings do not matter**. What matters is the user's
  general state.
- **Analysis must be more accurate and take longer** — not per-second, but
  **per-minute (1–2 minutes)** — so the chatbot can judge whether its approach
  suits this kind of state or whether it must change how it speaks.
- The user wants to see the user's reaction to *activities inside the
  conversation* — a game or a challenge — whether they are enjoying it,
  concentrating, and how they engage with it.
- Body-tracking (skeleton) is a **separate camera capability** from the face
  engine, and is part of a game that follows body movement.

### Answers already given

| Question | Answer |
|---|---|
| Does the user see their own emotion reading? | **No** — it does not help them |
| Calibration approach | Silent learning from early sessions (agent's proposal, accepted) |
| When does the camera run? | **Both**: activity-bound windows *and* when the bot suspects withdrawal |
| Who writes the bot? | The user, alone. Degree project. |
| Deliverable | Working app **plus** results and reports |

### Explicitly uncertain

The user stated plainly that details of the report and final submission **may
change**, including the two answers above about scope and deliverables. This is
why side-features are being built first: body tracking is useful whatever the
final report looks like.

**Do not treat requirements 1 and 2 as settled.** Ask before building anything
that depends on them.

---

## 3. Repository layout

The user's instruction: **every separate idea in its own folder, so that if one
breaks it does not break everything.**

```
C:\dev\psybot\
  ├─ AGENTS.md          this file
  ├─ engine_body\       body / skeleton tracking  ← start here
  ├─ engine_face\       face + affect, ported from mood_engine
  ├─ app\               the Flutter app
  └─ docs\              measurements, decisions, report material
```

Keep that separation. `engine_body` must not import from `engine_face`, and
neither may import from `app`. They meet only at a defined interface, so a
failure in one cannot take down the other.

The existing desktop project stays where it is at `C:\dev\mood_engine` — it is
**not** to be modified from here. It is a reference and a source of proven code.

---

## 4. What carries over from the desktop engine

`C:\dev\mood_engine` is a working Python + Flutter desktop system. It is a
source of *measured* answers, and several of its components are worth porting.

### Worth keeping

`engine/core/affect.py` already computes exactly what the chatbot needs, and
this was verified in the source:

| Signal | Range | Built from |
|---|---|---|
| `engagement` | 0..1 | attention × expressiveness |
| `tension` | 0..1 | brow knit, jaw and lip tightening |
| `fatigue` | 0..1 | blink rate, eye aperture, head droop |
| `volatility` | 0..1 | how much affect moves |
| `duchenne` | 0..1 | whether a smile reaches the eyes |
| `attention` | 0..1 | gaze direction toward the screen |
| `blink_rate` | per minute | — |

These are more useful to a chatbot than an emotion label, and more defensible.

Also worth porting in spirit: the guided-calibration idea, the personal
baseline correction, and the periodic session report.

### To be deleted for mobile

| Component | Measured cost | Why it goes |
|---|---|---|
| Face recognition (identity) | **~100 ms/frame** | One phone, one user |
| Age/gender estimation | ~15 ms/frame | The bot knows from sign-up |
| Person enrolment & matching | — | Meaningless with one user |
| MJPEG streaming + HTTP server | — | Everything is in-process |
| Snapshot storage | — | A privacy hazard in a mental-health app |

Removing identity alone halves the per-frame cost, and removes the cause of the
calibration-restart bug entirely (it was caused by `person_id` flickering).

### Cannot be ported

InsightFace / `buffalo_s` does not run acceptably on a phone. The realistic
substitutes are **ML Kit** or **MediaPipe** for detection and landmarks
(hardware-accelerated, free), and **TFLite** for expression. This is a rewrite
of the sensing layer, not a port.

---

## 5. Measured numbers — do not re-derive these

All obtained by running code on the user's machine (10-core CPU, 1280×720
webcam). They constrain the mobile design.

### Pipeline cost, desktop

| Stage | Cost |
|---|---|
| Capture | 32 ms/frame (31 fps) |
| Face detection alone | **19 ms** |
| + landmarks (106 pt) | +76 ms |
| + age/gender | +15 ms |
| + recognition | **+102 ms** |
| Emotion ONNX (224²) | 81 ms |
| **Full frame, desktop** | **~230 ms** |

Detector input size barely matters: 306 ms at 1280 px vs 184 ms at 640 px, and
it finds the same faces. Detection is *not* the bottleneck — the extra models
are.

### Emotion accuracy — the honest numbers

Measured on 1500 labelled faces across three corpora:

| Setup | Accuracy |
|---|---|
| Frozen model, BGR input (had a bug) | 58.6% |
| Frozen model, RGB input (fixed) | **61.3%** |
| On the highest-resolution corpus | 69.1% |
| On FER2013 (48 px) | 54.1% |

**61% is the ceiling of what is being sent to a chatbot.** Design accordingly.

### Three failed attempts to improve it

Recorded so they are not repeated:

1. **Linear head over 7 probs + 6 action units** — +4.7% on a random split,
   *lost* on 2 of 3 held-out sources. Refused by the guard.
2. **Same, with more and cleaner data** (61k images) — failed again.
3. **CNN trained on pixels** (20,747 aligned crops, 30 epochs) — AffectNet
   holdout 51.4% vs frozen 65.5% (**loss**); Cohn-Kanade 70.1% vs 66.7% (gain).
   Stopped by the user before finishing.

**The pattern is the conclusion:** every dataset has its own signature —
lighting, camera, acting style — and a model learns the signature instead of the
expression, then collapses on a new source. A phone camera is always a new
source.

This is precisely why personal calibration works where general training does
not: one source, no shift. Measured effect: **−37% in between-person variance.**

**Rule inherited from that work:** never ship a model that has not been scored
by leave-one-source-out. A gain on a random split is memorisation, because
near-duplicate frames land in both train and test.

---

## 6. Design decisions already agreed

### Per-minute analysis, not per-second

The desktop engine was constrained by the need to be live: `buffalo_s` over
`buffalo_l`, no ensembling, no test-time augmentation. All of those were the
price of instantaneous output.

A one-minute budget removes that constraint. Thirty samples instead of one, bad
frames filtered rather than accepted, median rather than a single reading. This
improves accuracy more than any training attempt did, because it attacks
variance — the dominant error source.

A second benefit: **the spread itself is a confidence measure.** Thirty
agreeing samples and thirty conflicting ones are different facts, and the
difference is information the bot should receive.

### What the engine sends the bot

**Not an emotion label.** A 61%-accurate classifier asserting "the user is sad"
to a mental-health bot is a quasi-clinical claim that is wrong four times in
ten. If the bot says "you seem sad today" and the user is not, that is
alienating, not empathic — in a therapeutic context that is a real harm.

Send instead:

```
engagement   0..1     attention x expressiveness
tension      0..1     brow knit, jaw/lip tightening
fatigue      0..1     blink rate, eye aperture, head droop
valence      -1..1    averaged over the window
trend        rising | falling | steady
confidence   0..1     from agreement between samples
```

**`trend` is the most valuable field.** It is a difference, and differences
cancel the personal baseline — the exact problem that defeated every training
attempt. "Engagement fell 30% during the game" is true regardless of the
classifier's absolute accuracy.

The bot uses these to adjust **tone and pacing**, never to comment on the
user's face. The effect should be felt, not announced.

### Camera windows — the resolved conflict

The user first said the camera should run "when the bot senses the user is
withdrawing", and separately that it should measure reactions to games and
challenges. These conflict:

- Withdrawal is *what you want to measure*, so it cannot also be the trigger to
  start measuring.
- If the bot already knows from the text, the camera adds nothing.
- Silent withdrawal — the most important case — would be missed entirely.
- Worst: a camera that only opens when a negative result is expected is a
  **biased instrument** that will keep confirming it.

Agreed resolution — **both**, but the activity-bound windows are primary:

| When | Duration | Why |
|---|---|---|
| During a game or challenge | length of the activity | The user's own priority |
| Session start | 60 s | A baseline to compare against |
| After a breathing/CBT exercise | 60 s | Did it actually help? |
| Bot suspects withdrawal | 60 s | Secondary, and must be flagged as such |

Windows tied to activities are **comparable** — same activity across users, or
across time for one user. That comparability is what makes the results section
of the report possible.

### Hard rule: the camera has no part in crisis detection

Not to trigger it, and not to suppress it. **Text alone decides.**

If the text says something concerning and the face reads "neutral", the text
wins — always. A 61%-accurate classifier must never stand between a person at
risk and an emergency line. Equally, a "tense" face must never open a crisis
path: false alarms in a mental-health app are harmful in themselves.

### Ethics — this is research with human participants

The camera needs explicit consent per window, or at minimum a permanently
visible indicator while it is on. An app that opens the camera "when it
suspects" something, without the user knowing, would fail an ethics review and
deserves to.

Cleanest: the user consents once to "measuring engagement during activities",
and sees an indicator whenever it runs.

### Battery

Continuous camera use costs 10–20% per hour. The windowed design solves this —
short windows rather than always-on.

---

## 7. Working agreements with the user

- **Language:** replies to the user in **Arabic**. Code comments, documentation
  and identifiers in **English**.
- The user asks for honest assessment over agreement. Negative results are
  reported plainly — three failed training attempts were, and that was right.
- **Measure before claiming.** Every performance or accuracy statement in this
  file came from running something.
- Ask permission before long-running or irreversible operations. The user
  appreciated being asked for all permissions up front in one batch.
- The user will sometimes say a requirement is uncertain. Believe them and ask.

### Environment hazards, learned the hard way

- Flutter lives at `C:\src\flutter\bin\flutter.bat` — **not** `C:\flutter`.
- PowerShell `Set-Content` and `.Replace` **corrupt UTF-8** files containing
  Arabic or `—`/`→`. Use `[IO.File]::WriteAllText($f,$c,(New-Object
  Text.UTF8Encoding $false))`, or the editing tools.
- .NET calls do not inherit `cd`. Always use absolute paths with
  `[IO.File]::ReadAllText`.
- Kill stray Python processes before any camera measurement: a held camera
  causes confusing timeouts.
- Timing tests that use the median are flaky on a busy machine — 722 ms under
  load vs 289 ms idle. Assert on the **best** sample: "can this machine do it?"
  is the real question.
- **No git repository** in the old project. Consider `git init` here early.

---

## 8. Body tracking — answered

The five forks in this section have been answered by the user. Full reasoning
and the four follow-up questions are in `docs/decisions-body.md`.

| Fork | Decision |
|---|---|
| Tracking stack | **ML Kit Pose Detection**, 33 landmarks |
| Body coverage | **Full body**, recorded |
| What is measured | **Body movement itself**; uses come later |
| Where it runs | **Fully native**; Dart receives landmarks only |
| Shared camera window with face | **No** — body only during the activity |
| Laptop simulation | Useful, not mandatory |
| Engine output | Raw landmarks **and** derived signals |
| Out-of-frame landmarks | Framing gate + quality labelling, nothing dropped |
| Test platform | **Android only** for now |

### The answer that reshaped the work

This section assumed body tracking was part of a game. **It is not.** The user's
words: *"the important thing is that it can detect the full body correctly.
There may be more than one use."*

So `engine_body` is a **sensing capability**. It contains no game logic, no
scoring, no target poses. Anything that plays is a consumer built on top.

This is the right shape for a project whose report may still change: a tracker
that works is useful under every version of it; a game is useful under one.

---

## 9. Status

### engine_body — partly built, partly unverified

| Part | State |
|---|---|
| Dart core: normalisation, angles, framing, smoothing, signals | **43 tests passing**, analyzer clean |
| Session recording and replay | Written and tested |
| Skeleton overlay painter | Written, widget-tested |
| Android layer: CameraX + ML Kit | **Runs.** 226 frames processed end to end on an emulator |
| Run on a real phone | **Not done** — emulator only so far |
| Device measurements | **Emulator only**, and not usable in the report |

The full pipeline works: camera opens, ML Kit loads its models, landmarks reach
Dart, the framing gate reports `noSubject` correctly with nobody in view.

**But this was an emulator with software rendering.** 6.2 fps is the emulator's
limit, not the design's. No number from it may enter the report as mobile
performance. See `docs/measurements.md`.

**Nothing has been seen tracking an actual body yet.** The emulator's virtual
camera never had a person in front of it.

### One real bug, already fixed

`DEFAULT_FRONT_CAMERA` threw *"unable to resolve a camera"*: the emulator
exposes one camera with `lensFacing = null`. Any phone without a front camera
fails the same way. `EngineBodyPlugin.resolveCamera` now falls back — requested
facing, then the opposite, then any camera.

**How it was found matters.** The failure was invisible: the app looked alive
and showed nothing, which is identical to an empty room. Diagnosing it required
`onError` on the stream and `debugPrint` to `adb logcat`. Flutter draws its own
widgets, so `uiautomator` sees no text — **on a device, a log line is the only
way to distinguish a working tracker from a dead one.** Keep the logging.

### The emulator is gone — develop on the laptop simulator

`engine_body_sim/` is a Python + MediaPipe program that runs the same
measurement model on the laptop webcam. **This is where body-tracking work
happens now.**

The Android emulator produced four real faults — skeleton beside the body,
camera cropped to one side, heavy stutter, camera often failing to open — and
all four were the emulator, not the design. It hands a *laptop* webcam to
Android while claiming a phone sensor at 90°, and runs inference and
compositing on one CPU.

| | Emulator | Simulator |
|---|---|---|
| Frame rate | 2.0 fps | **14.8 fps** |
| Body detected | occasional | **60 / 60 frames** |

Android Studio, the emulator, system images, AVD data and the NDK were deleted
(12.2 GB). **The base SDK and JDK were kept deliberately** — an APK can be
built the moment a real phone is connected, with no re-download.

**Rule learned, and it applies to the phone build too: mirror before
detecting.** Detecting on unmirrored pixels and flipping the image afterwards
puts landmarks in a different space from the display, which is exactly why the
skeleton floated beside the body.

### The laptop/phone split

ML Kit does not run on Windows. Rather than accept that as a blocker, the Dart
core is pure Dart — no Flutter, no plugin — so it runs in `flutter test` here.
Sessions recorded on a phone replay through the identical code, and a test
asserts replayed analysis matches live analysis exactly.

**The phone tests the sensor. The laptop tests the logic.**

### Next, in order

1. Connect an Android phone with USB debugging on, and run the example.
   Confirm a skeleton appears and follows the body.
2. Measure: fps, latency, battery per minute, `fast` against `accurate` on one
   recorded session.
3. Record a deliberately still subject to establish the sensor noise floor. The
   One-Euro constants and `stillnessReference` stay provisional until it exists.
4. Write every number into `docs/measurements.md`.

### Reference

`C:\dev\mood_engine` is complete and working: 178 Python tests, 101 Flutter
tests, clean analyzer, successful release build. Not to be modified from here.

### Environment additions

- **Android toolchain installed, but not in the usual places:**

  | Tool | Path |
  |---|---|
  | JDK 17 (Temurin) | `C:\dev\tools\jdk-17.0.20.1+1` |
  | Android SDK | `C:\dev\tools\android-sdk` |

  `flutter config` already points at both, and `JAVA_HOME` / `ANDROID_HOME` are
  set for the user. `flutter doctor` reports no issues.

- **Installers requiring admin fail silently-ish here.** `winget` returned exit
  code 1602 (UAC dismissed) for the Temurin MSI, and `--scope user` has no
  applicable installer. The working route was the **portable ZIP** for the JDK
  and **cmdline-tools** for the SDK, both unzipped to `C:\dev\tools`. Prefer
  that route again rather than fighting UAC.

- Accepting SDK licences non-interactively:
  `("y`n" * 60) | sdkmanager.bat --sdk_root=... --licenses`

- Android Studio **is** installed (`winget install Google.AndroidStudio`).

- **Emulator `psybot_body`** exists: Pixel 7, API 35, wired to the laptop
  webcam. Launch it with:

  ```
  emulator -avd psybot_body -camera-front webcam0 -camera-back webcam0 -gpu swiftshader_indirect
  ```

  `-gpu swiftshader_indirect` is not optional. With `-gpu host` the emulator
  froze hard enough that `adb` stopped responding and the processes had to be
  killed by PID.

  The AVD's `config.ini` must have `hw.camera.front=webcam0`; the default is
  `none`, and an emulated camera shows an animated fake scene with no body in
  it, so pose detection could never fire. **Edit the config before booting** —
  a running emulator ignores later changes.

- The first `flutter build apk` also pulled the NDK and Build-Tools 36, so it
  took about 11 minutes. Later builds take 15-40 s.

- `--dart-define=AUTOSTART=true` makes the example start tracking without a tap.
  Necessary for scripted runs: Flutter widgets are invisible to `uiautomator`,
  so `adb shell input tap` on a Flutter button is guesswork.

- **PowerShell `>` corrupts binary output.** `adb exec-out screencap -p > f.png`
  produces an unopenable file. Use `adb shell screencap -p /sdcard/f.png` then
  `adb pull`.

- `git init` has been done. Default branch `main`, first commit `ac07a48`.
- `.gitignore` excludes `*.jsonl`: session recordings are participant research
  data and must not enter version control.
