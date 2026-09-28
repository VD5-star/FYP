# Capture backend and frame size, measured

The simulator opened a window larger than the screen: the camera delivered
2560×1440 onto a 1920×1080 display, so only about a quarter of the image was
visible. Two assumptions in the code turned out to be wrong, and both had been
written as fact.

Machine: this laptop, Integrated Camera at index 1. Absolute numbers will not
transfer; the comparisons will.

## 1. The camera ignores every resolution it is asked for

`open_camera` set `CAP_PROP_FRAME_WIDTH/HEIGHT` to 1280×720 and never checked
whether it worked. It did not, in any combination of backend and pixel format:

| Asked | YUY2 via DSHOW | MJPG via DSHOW | MSMF |
|---|---|---|---|
| 640×480 | 2560×1440 | 2560×1440 | 2560×1440 |
| 960×540 | 2560×1440 | 2560×1440 | — |
| 1280×720 | 2560×1440 | 2560×1440 | 2560×1440 |
| 1920×1080 | 2560×1440 | 2560×1440 | 2560×1440 |
| 2560×1440 | 2560×1440 | 2560×1440 | 2560×1440 |

MJPG was tried because YUY2 is uncompressed and the USB bus alone can cap the
frame rate at high resolutions; the codec was set before the size, which is the
ordering that normally matters. It made no difference.

**The size request was dead code.** It has been removed rather than left in
place looking as though it did something, and the frame is resized after
capture instead.

## 2. MSMF is twice as fast as DSHOW here

The code preferred DirectShow, with a comment stating that MSMF "is slow to
open and has a habit of hanging on some laptop webcams". That was never
measured on this machine. Measured now:

| Backend | Frame rate | Open latency | Open + first frame | 8-index scan |
|---|---|---|---|---|
| DSHOW | 15.2 fps | 274–319 ms | 767–796 ms | 1651 ms |
| **MSMF** | **29.1 fps** | **52–134 ms** | 574–718 ms | 1472 ms |

MSMF wins on every axis, including the one it was avoided for. The backend
order is now MSMF → DSHOW → ANY, with DSHOW retained as a fallback because the
original claim may well be true of *other* cameras — it was simply not true of
this one.

**This doubled the capture frame rate on its own**, independently of any
resizing.

## 3. Downscaling before detection costs no angle accuracy

The frame has to shrink to fit the screen. The question is whether to detect on
the large frame and draw small, or shrink first and detect on the small frame.
Shrinking first keeps one coordinate space for landmarks, angles, overlay and
recording, which removes a whole class of "skeleton beside the body" defect —
but only if it does not cost accuracy.

Method: 180 frames of a real person (`real_squat.mp4`), upscaled to 2560×1440 to
match the camera, then reduced to each height with `INTER_AREA` and detected.
Left knee angle compared against detection at the full 1440.

| Detection height | mean abs diff | p95 | max |
|---|---|---|---|
| 1080 | 1.12° | 3.40° | 6.78° |
| **720** | **1.19°** | 4.29° | 5.78° |
| 540 | 1.33° | 4.69° | 6.48° |
| 480 | 1.28° | 4.22° | 5.87° |
| 360 | 1.22° | 4.16° | 6.81° |

**The difference does not grow as the frame shrinks.** 360 is no worse than
1080. A genuine loss of resolution would degrade monotonically; a flat
difference is resampling noise — the model landing on a slightly different
pixel, not a model that can no longer see.

Two controls were run first, because a difference this flat invites a wrong
conclusion:

- **Determinism.** Detecting the same input twice at the same size agrees to
  0.0000°. The spread above is therefore caused by the resizing, not by
  run-to-run variation.
- **A yardstick.** On a still subject at full resolution the same knee angle has
  σ = 0.54°, moving 0.23° between consecutive frames by itself. On the squat
  clip it moves 1.26° per frame on average. So the 1.19° downscale difference is
  about 0.9× the real per-frame motion — comparable to the signal, not lost in
  it, and larger than the still-subject noise floor.

That is the honest reading: **the downscale difference is not negligible in
absolute terms, but it does not worsen with scale**, which is what decides the
question. Detecting at 720 is not meaningfully different from detecting at 1080
or 360.

## 4. Shrinking does not speed up inference

The obvious reason to downscale would be speed. It is not the reason, and the
measurement says why:

| Height | read | resize | prep | **infer** | total | fps |
|---|---|---|---|---|---|---|
| 1440 | 11.05 | 0.00 | 9.95 | **12.74** | 33.74 | 29.6 |
| 1080 | 8.78 | 6.34 | 4.96 | **12.03** | 32.11 | 31.1 |
| 720 | 15.05 | 1.65 | 2.51 | **13.00** | 32.21 | 31.0 |
| 540 | 12.32 | 5.99 | 1.93 | **11.92** | 32.16 | 31.1 |
| 480 | 13.42 | 5.13 | 1.59 | **11.79** | 31.93 | 31.3 |

Milliseconds per frame, MSMF, 50 frames each.

**Inference costs ~12 ms regardless of input size**, because MediaPipe rescales
to its own fixed network input internally. Only the work *around* it scales:
BGR→RGB conversion drops from 9.95 ms to 2.51 ms, but the resize costs most of
that back.

So the end-to-end gain is real but small (29.6 → 31.0 fps). **The reason to
downscale is that the window fits the screen**, not speed. Claiming a speed win
here would be false.

End-to-end through the actual `main()` loop against the real camera: **35 fps
steady state** (median 28.5 ms between displayed frames, measured after the
first ten frames so camera warm-up and model construction are excluded).

## 5. Window sizing needs DPI awareness and the work area

Two Windows-specific traps, both silent:

- A process that has not declared DPI awareness is told the *scaled* screen
  size, and then has every window stretched by the scale factor. Sizing a window
  to the reported height would still overflow on a 150% display. Fixed by
  calling `SetProcessDpiAwareness(2)` before asking. This machine runs at 100%,
  so the fault would not have shown here — it would have shipped and broken
  elsewhere.
- The **work area** (1920×1032) is smaller than the screen (1920×1080) because
  of the taskbar. Fitting to width alone is the classic error and fails exactly
  here: 2560×1440 scaled to 1920 wide is 1080 tall, so the feet would sit behind
  the taskbar — on a tool whose job is watching feet. The fit is limited by both
  axes, with a 0.90 margin for the title bar and borders.

The window is created `WINDOW_NORMAL`, not the `WINDOW_AUTOSIZE` default:
AUTOSIZE locks the window to the image size and refuses to be resized, which is
what left no way to shrink the oversized preview by hand.

Resulting window for this camera: **1651×928**, inside the 1920×1032 work area.

## Defect found while reading

`last` held the fps timestamp, but the hold-exercise branch reassigned it to a
`Hold` object when reporting the previous hold. The next frame would then
subtract a `Hold` from a float. Only the three hold exercises (`plank`,
`stretch`, `balance`) reach that line, and only after a hold completes, which is
why it had never been seen. Renamed to `last_hold`.

## Settings that follow

- `CAPTURE_BACKENDS` = MSMF, DSHOW, ANY on Windows.
- No resolution request at capture; `working_size()` shrinks after, never
  enlarges.
- `DEFAULT_WORKING_HEIGHT` = 720, inside the 360–1080 band actually measured.
- `--height N` to override, `--height 0` or `--full-size` to disable.

## Not established

- All of this is one camera on one laptop. The DSHOW-vs-MSMF result in
  particular is known to go the other way on some hardware, which is why DSHOW
  is kept as a fallback rather than deleted.
- The downscale accuracy test could not be carried through to **rep counts**,
  which is the outcome that actually matters. The clip is a still photograph
  under rigid transforms, and the viewpoint gate correctly rejects all 180
  frames with "turn to face the camera", so the pipeline produces no reps to
  compare at any resolution. Consistent with the standing rule that **a still
  image proves tracking, not movement**. The angle comparison is what stands.
