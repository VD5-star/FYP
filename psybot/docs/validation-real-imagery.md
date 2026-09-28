# Validation on real photographic imagery

This is the first validation of the movement layer against real image data
rather than rendered shapes, and it settles the question Round 4 left open.

## Why this was necessary

`docs/research-round4-notes.md` records that the synthetic clips were useless
for testing anything downstream of the model: correlation between detected hip
height and the driving animation signal was **−0.04** — MediaPipe never tracked
the rendered figure at all. Every accuracy claim made on those clips had to be
withdrawn.

## Method

Source: MediaPipe's official test photograph of a real person, moved along
exactly known trajectories.

This keeps what the rendered figures lacked — **photographic texture, shading,
clothing, real edges** — while retaining exact ground truth. It does not
reproduce a person actually moving their limbs, and that limitation is stated
below.

Sensor noise (σ = 2.5 grey levels) and sub-pixel camera shake (σ = 0.4 px) were
added to every frame.

## Result 1 — the model tracks real texture

| Clip | Correlation, tracked height vs true height |
|---|---|
| Rendered figure (Round 2) | **−0.04** |
| Real photograph | **0.9970** |

The difference is entirely the imagery. **The tracking pipeline was never the
problem.**

## Result 2 — jump detection

A ballistic trajectory with a true peak of 0.30 m and true flight of 0.495 s:

| | Measured | True |
|---|---|---|
| Flight time | 0.467 s | 0.495 s |
| Fitted gravity ratio | 1.07 | 1.00 |
| Detections | 75 / 75 frames | — |

Flight-time error is **28 ms**, which is *better* than the +61 ms bias reported
in the published validation at 240 fps — though that study measured real human
jumps with real take-off ambiguity, so this is not a like-for-like comparison
and should not be quoted as beating it.

The fitted gravity ratio of 1.07 confirms the free-fall check works on real
detections and not only on ideal coordinates.

## Result 3 — the two hard rejections

Both on the same real photographic texture:

| Movement | Result |
|---|---|
| Tiptoe rise, 12 cm, no free-fall phase | **correctly rejected** |
| Postural sway, 1.5 cm | **correctly rejected** |

Tiptoes is the discrimination the literature does not cover. It is decided by
physics: during flight the only force is gravity, so the centre of mass traces
a parabola with acceleration g. A tiptoe rise has no such phase.

## Result 4 — a real defect this found

Running the full pipeline on the photograph exposed a fault no synthetic test
could have:

**The viewpoint gate reported 64° of yaw and refused to analyse a subject whose
true yaw was 6.7°.** The person was facing the camera and the app would have
told them to turn around.

Ground truth came from MediaPipe's 3D world landmarks. The cause was
anatomical: the nose follows the *head*, and a person can turn their head while
their torso faces the camera squarely.

Fixed by `torso_yaw_degrees`, which uses the shoulder line in 3D. Verified: 6.7°
on the same photograph, and the session now accepts the frame. The gate remains
strict for a genuinely side-on torso.

## What this does and does not establish

**Established:**

- MediaPipe tracks real photographic texture reliably (0.997 correlation)
- Centre-of-mass tracking, jump detection and the free-fall check work on real
  detections
- Tiptoes and sway are rejected on real imagery
- The angle layer produces sensible values on a real pose: eight joints at 0.99+
  confidence, and it correctly reported an asymmetric stance as asymmetric

**Not established:**

- **A person actually moving their limbs.** The subject is rigid and
  translated. Real movement brings self-occlusion, motion blur and
  foreshortening, and repetition counting therefore remains unvalidated on real
  imagery.
- **More than one body, one outfit, one lighting condition, one camera.**
- **Any phone.** All measurement is on a laptop CPU.

The next measurement, when the camera arrives: a real person performing real
squats, which is the only thing that can validate the rep counter end to end.
