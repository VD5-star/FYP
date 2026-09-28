# The jump glitch: measured, and what could not be fixed

The user reported a glitch when jumping. This is what it was, how it was
reproduced, and why the obvious fix was wrong.

## Reproduction

`testdata/real_jump.mp4` did **not** reproduce it: all 75 frames come back
usable. That clip is a photograph under rigid translation, and the standing rule
applies — *a still image proves tracking, not movement*
(`docs/validation-limits.md`).

So the jump was built in **landmark space**, which
`docs/research-round4-notes.md` explicitly sanctions: the modules under test are
pure functions of landmark input, so synthetic landmark sequences test the logic
exactly. Synthetic *video* remains banned.

The fixture derives the whole motion from one quantity — hip height — so every
phase is continuous by construction. Knee bend and hip height are linked as they
are in a body, flight is a true parabola, and push-off and landing are
raised-cosine so they start and end at zero velocity. Phase durations come from
the countermovement-jump literature: countermovement 250–400 ms, push-off
200–300 ms, landing absorption 100–200 ms.

An early version of this fixture stepped straight into a 0.33 knee bend on the
first landing frame. That is a teleport and would rightly trip any guard — **the
fixture was wrong, not the code**, and the first "defect" it found was discarded.

## The defect

`SubjectSwitchDetector` fired mid-jump. The session then reset the rep counter
and discarded the repetition in progress.

The `hip_rate` limit was 9.0, documented as leaving *"generous headroom for a
jump, which moves the hips faster than a squat does"*. That was reasoned, not
measured. Measured:

| Movement | peak hip rate |
|---|---|
| normal countermovement jump | 6.14 /s |
| deep crouch (bend 0.85) | 10.28 /s |
| **explosive jump, 133 ms push-off** | **13.92 /s** |

The threshold sat *inside* the population it was supposed to be above.

## Why raising the number is not enough

Measuring the other population shows the two overlap:

| | hip rate |
|---|---|
| explosive jump, push-off | 13.92 /s |
| switch to same-build person half a torso away | 15.00 /s |

**No threshold on hip speed separates them.** Two further discriminators were
tried and rejected on measurement:

- **Direction.** A jump is vertical (vertical share 0.86 on its fastest frames)
  and a person standing elsewhere is displaced horizontally. But a real crouch
  moves the hips *backwards* — the fixture's own anatomy produces 5.4–7.2 /s of
  horizontal hip motion — so horizontal speed is not exclusive to a switch.
- **Motion prediction.** Constant-velocity residual fails because a jump *is* an
  acceleration (worst real 0.35 torso-lengths, weakest switch 0.15 — inverted).
  Constant-acceleration residual fails too (0.54 vs 0.15).

## A persistence rule was tried and rejected by test

The next attempt required the anomaly to hold for **two consecutive frames**, on
the reasoning that a jump's peak is transient while a switch persists.

A test disproved it immediately. A genuine switch is anomalous on exactly **one**
frame — the transition — and the substituted person is perfectly steady
thereafter. "Two consecutive anomalous frames" therefore never occurs, and the
guard stopped detecting switches at all.

The reasoning had confused *the person persisting* with *the rate staying high*.
Only the first is true. This is recorded because the idea is superficially
convincing and would otherwise be tried again.

## What was actually changed

Both thresholds moved above every measured real movement.

Population: **399 physiologically-timed movements** at 30 fps — jumps of 5–75 cm
crossed with push-off 133–300 ms, absorption 100–200 ms and crouch depths
0.4–0.85, plus squats at five depths and three tempos. Landmark noise at the
**1.09 px measured on a real human** (`docs/measurements-real-human.md`).

| Noise | worst torso rate | worst hip rate |
|---|---|---|
| none | 1.97 /s | 17.38 /s |
| **1.09 px (measured)** | **2.66 /s** | **17.50 /s** |
| 3 px (stress) | 3.92 /s | 17.72 /s |

| | torso | hip | false positives | switches caught |
|---|---|---|---|---|
| before | 7.0 | 9.0 | **186 / 399** | 31 / 35 |
| **after** | **4.0** | **20.0** | **0 / 399** | 26 / 35 |

Torso rate went *down* as well: 7.0 was above every build difference under 23%,
so it caught almost nothing and the hip test was doing all the work — while
firing on jumps. 4.0 sits between the worst real movement (2.66) and a 15% build
change (4.50).

The trade is explicit: **186 false positives removed, 5 detections lost.** The
five lost are all small-build-difference, small-displacement cases.

## What this still cannot do — stated, not tuned away

A switch to a person of **the same build standing within half a torso** of the
previous subject is not detectable from these quantities. It gives a hip rate of
15 /s — inside the jump population — and no torso change at all.

Lowering the thresholds until that case is caught is exactly what produced the
reported glitch. It is a limit of the measurement, not a defect.

## The other guards are clean

The reliability monitor and bone-length monitor were checked on the same
sequences, since a jump moves landmarks fast and they reject "teleports":

```
gentle 40 cm jump     89 / 89 frames usable
explosive 40 cm jump  84 / 84 frames usable
```

Neither fires. The subject-switch guard was the only one at fault.

## Not established

- All of this is synthetic landmark sequences. They test the *logic* exactly and
  assume the landmarks are correct. **Rep counting on a real moving human
  remains the one unvalidated thing**, exactly as before.
- The 26/35 detection rate is against a sampled grid of switches, not a measured
  distribution of how people actually walk into frame.
- The `PoseSmoother` lag measurement is **still not done**. It was attempted
  against the live camera, but with nobody in front of it there is no tracked
  signal to measure lag against. It needs a person, and the filter stays
  configurable and off by default until then.
