# engine_body_sim — laptop simulator

Body tracking on the laptop webcam, with MediaPipe Pose. One Python window:
camera, skeleton, joint angles, framing state.

## Why this exists

The Android emulator was abandoned. It was not a design problem — it was the
emulator pretending to be hardware it is not:

| Reported fault | Cause |
|---|---|
| Skeleton beside the body, vertical while the user was horizontal | The emulator passes a **laptop** webcam to Android while claiming a phone sensor mounted at 90°. The landmarks and the image ended up in different spaces. |
| Camera pushed to the far right | Same rotation mismatch, plus cropping. |
| Heavy stutter | Inference and compositing on the same CPU, no GPU. **2.0 fps.** |
| Camera often failed to open | Emulated camera stack; the emulator also hung hard enough to need killing by PID. |

Measured here, on the same laptop, with a real camera:

| | Emulator | This simulator |
|---|---|---|
| Frame rate | 2.0 fps | **14.8 fps** |
| Body detected | occasionally, `partial` | **60 / 60 frames** |
| Camera open time | often failed | **1.1 s** |

**7× the frame rate**, and nothing is being emulated.

## Run

```powershell
cd C:\dev\psybot\engine_body_sim
python pose_sim.py                      # live
python pose_sim.py --record session.jsonl
python pose_sim.py --camera 1           # another camera
python pose_sim.py --no-mirror
```

Keys: `q` quit, `s` save a still.

## The fix for "the skeleton is not on my body"

**Mirror first, then detect.**

The emulator detected on unmirrored pixels and then flipped the image for
display, which put the landmarks in a different space from what was on screen —
so the figure landed on the wrong side and moved the wrong way.

Here the frame is mirrored *before* it reaches MediaPipe, so there is only ever
one coordinate space and the skeleton cannot drift off the body.

## What it shares with the phone engine

The measurement model is the same, deliberately:

- MediaPipe's 33 landmarks — the same set ML Kit produces
- the torso frame: origin at mid-hip, one unit from mid-hip to mid-shoulder
- joint angles measured against the adjoining segment
- the framing gate: `full` / `upperBodyOnly` / `partial` / `noSubject`
- One-Euro smoothing on the drawn skeleton, no interpolation between detections

Recordings are written in the **identical JSONL schema** `SessionReplay` reads,
so a session captured here replays through the real Dart engine core unchanged.

## What it does not claim

`14.8 fps` is a measurement of **this laptop**, not of a phone. A phone has an
NPU and a GPU delegate and will behave differently — better or worse is not
known, and must not be asserted until measured on a device.

Nothing here reads a psychological state. Movement is geometry: joint angles,
trunk lean, framing quality. Any mapping from those to a state has to be
measured first.
