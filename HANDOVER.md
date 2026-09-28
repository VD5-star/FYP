# PsyBot — Handover before formatting the machine

Written: 2026-09-21
Machine: Windows, user `jubra`

---

## 0. READ THIS FIRST — what you are about to lose

**Nothing in `C:\dev` is backed up anywhere. There is no git remote on any
folder. If you format now, all of it is gone.**

| Folder | Size | Git repo? | Remote? |
|---|---:|---|---|
| `C:\dev\games` | 2,537 MB | no | no |
| `C:\dev\mood_engine` | 3,494 MB | no | no |
| `C:\dev\psybot` | 185 MB | **yes** | **no** (86 uncommitted files) |
| `C:\dev\psybot_dart` | 2,984 MB | no | no |
| **total** | **12,363 MB** | | |

Your Desktop is inside OneDrive (`C:\Users\jubra\OneDrive\...`) so it syncs to
the cloud. `C:\dev` is **not** in OneDrive and does **not** sync.

### What you must copy off this machine

Most of the 12.4 GB is regenerable junk. Skip these:

| Regenerable — do NOT copy | Size |
|---|---:|
| `psybot_dart\apps\games\build` | 2,388 MB |
| `psybot_dart\apps\games\.dart_tool` | 197 MB |
| `psybot_dart\packages\engine_body\build` | 47 MB |
| `games\flutter_games\build` | 2,348 MB |
| `games\flutter_games\.dart_tool` | 179 MB |
| `mood_engine\.venv` | 1,466 MB |

That is ~6.6 GB of rebuildable files. What genuinely matters:

1. **All source code** — 125 `.dart` files, 106 `.py` files. Tiny, a few MB.
2. **`mood_engine\assets`** (45 MB) — the two ONNX models
   `enet_b0_8_best_vgaf.onnx` and `enet_b2_8.onnx`. These are downloaded
   models; re-obtainable but slow.
3. **`mood_engine\datasets`** (748 MB), **`mood_engine\data`** (519 MB),
   **`mood_engine\app`** (714 MB) — decide deliberately. If the datasets came
   from a public source you can re-download them. If any of it is **your own
   collected face data, it is irreplaceable.** Check before you wipe.
4. **`C:\dev\psybot\.git`** — carries the only real history in the project.

Suggested: copy everything except the six regenerable folders above to an
external drive or OneDrive. That is roughly 5.8 GB.

---

## 1. What this project is

PsyBot has two halves that are now deliberately separate:

- **Desktop version — Python.** `C:\dev\games`, `C:\dev\mood_engine`,
  `C:\dev\psybot`. **Untouched and kept on purpose.** This is the reference
  implementation and stays as the PC version.
- **Mobile version — Dart/Flutter.** `C:\dev\psybot_dart`. Android only
  (Windows scaffolding was deliberately deleted).

The work of the last sessions was porting the Python to Dart for mobile, not
replacing it.

---

## 2. Where we stopped — current state

Everything is green as of the last run:

| Package | Tests | Analyzer |
|---|---:|---|
| `packages\engine_body` | 304 pass | clean |
| `packages\mood_engine` | 168 pass | clean |
| `apps\games` | 214 pass | clean |
| **total** | **686 pass** | clean from workspace root |

APK builds: `flutter build apk --debug` → 221 MB debug APK.

### The mobile workspace layout

```
C:\dev\psybot_dart\
  pubspec.yaml               workspace root, 3 members
  packages\engine_body\      body/skeleton tracking, Android plugin
  packages\mood_engine\      face affect/mood/baseline logic, pure Dart
  apps\games\                the three games + main.dart
```

All three have `resolution: workspace` so `flutter analyze` from the root
checks everything at once. Note: `flutter test` from the root does **not**
aggregate — that is normal Flutter behaviour. Run tests per package.

---

## 3. What was completed, in order

### 3.1 Python games (desktop) — finished earlier
- `jigsaw`: explicit PICK/SETUP/GAME pages, `reshuffle(seed)` that re-scatters
  only loose pieces, circular stop button. Real performance fix: `canvas[:] =
  BG` was 5.5 ms of a 6.8 ms frame; switching to `cv2.rectangle` took it to
  **0.71 ms from 6.77 ms**. 5 checks pass.
- `attend`: converted to the same single-button style. Opening the button now
  genuinely stops the session and silences audio. 4 checks pass.

### 3.2 Dart port of the three games
- `lib/core/` (610 lines): zones, paint, synth, audio.
- `lib/jigsaw/` (3,488 lines, 62 tests). Fixed a 4.8 s main-thread freeze by
  moving generation and cutting into isolates.
- `lib/reach/` (2,997 lines). Measured parity with Python: bonus timed **150**
  / calm **0**, hits **899**, `bonusTaken` 149, `bonusTime` 447.0.
- `lib/attend/` (2,200 lines, 41 tests). Found three real bugs while porting:
  bowl strikes genuinely overlapped (`dn=105840` vs gap `104857`), a wrong
  Hilbert envelope gave `deep` roughness 16.0 (6.4 after fixing), and my own
  tests were measuring the wrong signal.

### 3.3 mood_engine ported to Dart
16 files in `packages\mood_engine\lib\src\`: config, affect, baseline, mood,
session, calibration, attributes, emotion, recognizer, detector, database,
engine, geometry, maths, sqflite_store.

Models sit behind interfaces (`EmotionModel` + `FakeEmotionModel`,
`FaceEmbeddingModel`, `MoodStore` with an in-memory implementation) so all
168 tests run with no camera, no model and no device.

Scope decision that was followed: **logic only**. Training code
(torch/sklearn/insightface) and the fastapi server stay Python.

### 3.4 reach unified onto engine_body — the hard one

`reach` had its own copies of `anchor.dart`, `occlusion.dart`,
`skeleton.dart`, `smooth.dart` (~604 lines) duplicating the shared package.

Before touching anything I measured, and found a real conflict:

- The algorithms were **identical** — comparing `JointAnchor` across 6,600
  points with the same input gave a difference of exactly **0.000**.
- But `reach` worked in **pixels** and `engine_body` in **normalised 0..1**.
- One-Euro's `beta * |dx|` term is **scale-dependent**, so the same constants
  behave differently: 86.5 % jitter retained in pixel space vs 83.2 %
  normalised.
- Worse, per-axis normalisation (x÷width, y÷height) **distorts radial geometry
  by 33.3 %** on a 480×640 frame. Targets would become ellipses and "arm's
  length" sideways would differ from upward. Measured: isotropic scaling gives
  exactly **0.0 %** distortion.

**Decision taken (yours): isotropic normalisation with a conversion
boundary.** `reach` divides both axes by the frame *height*, so geometry is
undistorted; conversion to per-axis 0..1 happens only at the engine_body call.

Result: the four duplicate files were deleted, `bridge.dart` (117 lines) was
added as the boundary, `body.dart` was rewritten (106→150 lines), and the
golden gameplay numbers **150 / 899 / 149 / 447.0 stayed exactly the same**,
proving gameplay was preserved.

Beta was re-derived by measurement, not guessed: `beta × 720` (the frame unit)
reproduces pixel behaviour exactly — 103.1 % retention in both cases.

Two changes were needed inside `engine_body` (it serves other projects):
- `OcclusionTracker` / `OcclusionReport`: `believeThreshold` is now injectable
  (was hardcoded 0.5; reach needs 0.2).
- `PoseSmoother`: added `gapReset` (100 ms) and `dropFilterBelowThreshold`.
  Both existed in reach and were missing from the package; dropping them would
  have silently changed behaviour. Defaults preserve old behaviour and all 304
  tests stayed green.

### 3.5 Mobile-only cleanup + three real bugs found

- Deleted `apps\games\windows\` (19 files). No `windows/`, `linux/`, `macos/`
  or `web/` folders remain.
- **Camera permission was completely missing.** `AndroidManifest.xml` had no
  `android.permission.CAMERA` at all, although `reach` opens the camera. The
  game would have failed on any real device. Added the permission plus
  `uses-feature` entries with `required="false"`. Verified it is present in the
  merged manifest inside the built APK, and confirmed the camera plugin
  requests it at runtime via `CameraPermissionsManager`.
- **The workspace was never actually functional.** The root declared
  `workspace:` but no member had `resolution: workspace`, so `flutter analyze`
  from the root always failed. Fixed in all three packages.
- The symlink / Developer Mode warning turned out to be caused by the Windows
  scaffolding, not by a machine restriction. It disappeared when
  `apps\games\windows\` was deleted. **No admin rights are needed.**

### 3.6 Camera picker (last feature added)

New file `apps\games\lib\reach\cameras.dart` — pure logic, no platform
dependency, so it is testable with no device. Provides `CameraChoice`,
front-camera-by-default, cyclic switching, numbering for same-facing cameras
(`back 1` / `back 2`), and recovery of a remembered camera by name.

The button appears in the reach menu **only when more than one camera exists**.
Pressing it moves to the next camera, reopens it, and **resets the tracker** so
no body pose leaks across cameras.

Also fixed: `mirror` was hardcoded `true`. Mirroring now follows the actual
camera (`mirroring`) — front mirrored, back not. Without this the rear camera
would have produced inverted movement.

While writing the layout tests, they caught two genuine bugs: at 640×480 the
button fell **off-screen** (517 > 480), and after clamping it **covered the
`begin` button** (408 < 453). Root cause was that the menu stack does not fit
in 480 px of height. Fixed at the root — the menu now computes its required
height and lifts itself when the camera row is present.

24 camera tests added (214 total in `apps\games`).

---

## 4. Where we are stopped right now

**The next step is testing the APK on a real phone.**

This is the single most important open item, and it is riskier than it looks:
because the camera permission was missing entirely, **the camera / ML Kit path
has never run even once.** Everything proven so far is logic, layout and
permissions being correct — *not* that the camera actually sees your body.

Specifically unverified on real hardware:
- camera opens and streams at all
- ML Kit pose detection produces landmarks
- the isotropic coordinate conversion looks right on screen
- the camera picker switches cameras without crashing
- audio actually sounds correct (it was only ever verified mathematically;
  `flutter_soloud` does not run in headless tests)
- rep counting against a real moving human — the oldest open item in the
  project

To install: `cd C:\dev\psybot_dart\apps\games` then
`flutter build apk --release` (the 221 MB figure is the *debug* build; release
is far smaller). The APK lands in
`build\app\outputs\flutter-apk\`.

---

## 5. Known gaps and honest caveats

- `flutter test` from the workspace root does not aggregate. Run per package.
- reach test count went 92 → 85 during unification: 14 tests that poked at the
  internals of the deleted files were replaced with 7 behavioural ones. Some
  fine-grained internal checks were genuinely lost (`OneEuro.alpha` maths, the
  1e-9 anchor precision) because `engine_body` does not expose those.
- `apps\games\lib\targets.dart` and `draw.dart` are still entirely in pixels.
  This is deliberate — conversion happens inside `body.dart` only, which is
  exactly why the golden numbers survived.
- jigsaw images are not pixel-identical to Python (different RNG, linear vs
  cubic scaling). Structure, seeds and the no-flat-piece guarantee are kept.
- Latent issue carried over: `occlusion` uses `skeleton.torso_length` while
  `anchor` is fed `movement.torso_length`. Measured at 0.0 % divergence over
  1,334 frames, so harmless today, but they should be unified.
- `ios/` scaffolding does not exist. The code supports iOS logically
  (`Platform.isIOS` is handled) but generating and building it needs a Mac.
- Timing measurements on this machine are genuinely noisy — the same code gave
  6.9 ms and 19.5 ms. Timing checks therefore take the best of five runs.
- `google_mlkit_pose_detection` 0.16.1 supports Android/iOS only. There is no
  Windows body tracking, which is part of why the desktop build was dropped.

---

## 6. Environment — what to reinstall after the format

| Tool | Version | Path used |
|---|---|---|
| Flutter | 3.47.2 | `C:\src\flutter\bin\flutter.bat` |
| Dart | 3.13.2 | `C:\src\flutter\bin\dart.bat` |
| Python (desktop games) | 3.14 | `C:\Program Files\Python314\python.exe` |
| Python (mood_engine) | 3.11 + pytest | `C:\dev\mood_engine\.venv\Scripts\python.exe` |
| Android SDK | — | `C:\dev\tools\android-sdk` |
| JDK | 17.0.20.1+1 | `C:\dev\tools\jdk-17.0.20.1+1` |

`C:\dev\tools` contains only the Android SDK and JDK — no project code. It is
re-installable, so it does not need backing up.

After restoring: `flutter pub get` in `C:\dev\psybot_dart`, then per-package
`flutter test`. Recreate `mood_engine\.venv` with Python 3.11 and reinstall
from the project's requirements.

---

## 7. Project rules that were being followed

These are your standing instructions, recorded so they survive the format:

- **Zero comments in code.** No `//`, no `///`, no `#`, no TODOs. Code only.
- **No `.md` files or documentation** unless explicitly asked (this file was
  explicitly requested).
- Replies in Arabic, code in English, and keep replies short.
- Write only: where we stopped, what the problems are, whether fixes exist.
- Do not start coding unless asked; do not add unrequested improvements.
- Do not re-run tests after the user confirms things work.
- Answer direct questions directly, without running tools.
- Project maxim: **"measure before you claim."** Priority is speed and
  responsiveness.
