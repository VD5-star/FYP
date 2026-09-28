# engine_body

Body / skeleton tracking sensing layer for PsyBot.

**This is a sensing capability, not a game.** It detects the body and publishes
what it sees. Games, challenges and any other consumer are built on top of it,
in their own packages. See `../docs/decisions-body.md` for why.

## Status

| Part | State |
|---|---|
| Dart core — normalisation, angles, framing, smoothing, signals | **43 tests passing**, analyzer clean |
| Session recording and replay | Written and tested |
| Skeleton overlay painter | Written, widget-tested |
| Android layer — CameraX + ML Kit | **Compiles**, debug APK builds, 3 JVM tests pass |
| Run on a device | **Never** — no phone has been connected |
| Device measurements | **None taken** |

### What compiling does and does not prove

The Kotlin layer is type-correct and links against CameraX and ML Kit.

It has never been **run**. The camera has not been opened, no skeleton has been
drawn, and no landmark has ever come out of ML Kit. Threading, camera lifecycle
and image rotation are all unexercised, and they are where the defects will be.

No frame-rate, latency or accuracy number appears anywhere in this package,
because none has been measured. `AGENTS.md` requires measurement before a claim,
and that rule is not suspended because a number would be convenient.

## Structure

```
lib/
  src/core/       pure Dart. No Flutter, no plugin, no camera.
  src/platform/   the Android channel. Cannot run on Windows.
  src/ui/         skeleton overlay painter.
android/          CameraX + ML Kit. Emits landmarks only.
test/             runs on Windows with no device.
```

`src/core` is deliberately free of Flutter and plugin imports. That is what lets
the analysis be developed and tested on a laptop that cannot run ML Kit.

## Working without a phone

ML Kit has no Windows implementation, so a live skeleton on the laptop is
impossible with this stack. The workaround is built in:

1. Record a session on an Android device with `SessionRecorder`.
2. Copy the `.jsonl` file to the laptop.
3. Replay it through `SessionReplay`, which feeds the **identical** core.

A test asserts that replayed analysis matches live analysis exactly. Without
that guarantee, tuning done on the laptop would not transfer to the device.

**The phone tests the sensor. The laptop tests the logic.**

## The measurement model

Every derived value is computed in **torso units**, not pixels:

- origin — the mid-hip point
- one unit — the mid-hip to mid-shoulder distance
- +y — upwards, unlike image coordinates

So a raised wrist reads the same for a tall person, a short person, someone
close to the camera and someone across the room. This is tested directly.

It does **not** correct for a subject turning away from the camera. An arm
extended towards the lens is foreshortened and reads short. ML Kit's `z` could
in principle help, but ML Kit documents it as experimental and it has not been
validated here, so nothing depends on it.

## Confidence is never hidden

ML Kit returns all 33 landmarks whether or not it can see them. Off-screen legs
are not omitted — they are inferred, with a low `inFrameLikelihood`. Code that
ignores this treats invented coordinates as measurements.

So every frame carries a `FramingReport`:

| Framing | Meaning |
|---|---|
| `full` | Torso, arms and legs all reliable |
| `upperBodyOnly` | Seated user; torso and arms valid, legs are not |
| `partial` | Torso unreadable — no coordinate system, no derived signals |
| `noSubject` | Nobody detected |

Unmeasurable values return `null`, never a plausible-looking default.

One caveat, stated because it changes how a threshold reads: ML Kit publishes
only **one** confidence per landmark, and the Android layer writes it to both
`likelihood` and `inFrameLikelihood`. The framing gate multiplies them, so on
Android it squares a single value. A `groupThreshold` of 0.6 therefore demands
about 0.77 from ML Kit — stricter than it looks, which is the safe direction,
but not directly comparable to an ML Kit figure quoted elsewhere.

## What this package will not claim

`movementEnergy` is displacement in torso units per second. It is **not**
agitation. `stillness` is **not** calm — a rigid, motionless subject scores the
same whether they are relaxed or frozen. `postureOpenness` measures wrist
distance from the midline, and someone holding a phone has their arms in close
for reasons that have nothing to do with their state.

This restraint is inherited, not invented. The face engine's classifier is 61%
accurate and `AGENTS.md` forbids it from asserting an emotion to a mental-health
bot. Body movement has **no measured accuracy here at all**, so it has strictly
less licence to interpret, not more.

Any mapping from these numbers to a psychological state must be measured first,
and scored leave-one-source-out, per the rule inherited from three failed
training attempts recorded in `../docs/measurements.md`.

## Consent

The plugin **checks** camera permission and never requests it. The app layer
asks, because only the app can explain why a mental-health assistant wants a
camera. The example app shows a permanent indicator while the camera is on, as
`AGENTS.md` requires.

Recordings contain coordinates only. They must never be extended to store
images, and `.gitignore` keeps `.jsonl` files out of version control: they are
participant research data, not source.

## Building

The toolchain lives outside the usual locations, because installers needing
admin rights could not run here:

```
JAVA_HOME    C:\dev\tools\jdk-17.0.20.1+1
ANDROID_HOME C:\dev\tools\android-sdk
```

Both are already set for the user and registered with `flutter config`.

```powershell
# Dart core - runs on Windows, no device needed
flutter test

# Android layer
cd example
flutter build apk --debug

# Kotlin JVM tests
cd example/android
./gradlew :engine_body:testDebugUnitTest
```

## Next steps

1. Connect an Android phone with USB debugging enabled; `flutter run` in
   `example/` and confirm a skeleton appears and tracks the body.
2. Measure: frames per second, latency, battery per minute, `fast` against
   `accurate` on the same recorded session.
3. Record a deliberately still subject to find the sensor noise floor — until
   that exists, the One-Euro constants and `stillnessReference` stay provisional.
4. Write the numbers into `../docs/measurements.md`.
