from __future__ import annotations

import math
import sys

import numpy as np

import movement as m
from pose_sim import IDX

def make_pose(knee_bend: float = 0.0, arm_raise: float = 0.0,
              rise: float = 0.0, scale: float = 1.0,
              centre_x: float = 360.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a plausible 33-landmark pose in pixel coordinates.

    knee_bend  0 standing, 1 deep squat
    arm_raise  0 arms down, 1 arms horizontal
    rise       vertical offset in pixels, negative is upwards (a jump)
    scale      body size, standing in for distance from the camera
    """
    pts = np.zeros((33, 2), dtype=np.float64)
    vis = np.ones(33, dtype=np.float64)

    def place(name: str, x: float, y: float) -> None:
        pts[IDX[name]] = (centre_x + x * scale, (y + rise) * scale)

    shoulder_y = 300.0
    torso_length_px = 200.0
    femur = 200.0
    tibia = 180.0

    knee_angle_rad = math.radians(180 - knee_bend * 105)
    shin_lean = knee_bend * 0.45
    ankle_y = 880.0

    knee_y = ankle_y - tibia * math.cos(shin_lean)
    knee_forward = tibia * math.sin(shin_lean)
    thigh_lean = math.pi - knee_angle_rad - shin_lean
    hip_y = knee_y - femur * math.cos(thigh_lean)
    hip_back = knee_forward - femur * math.sin(thigh_lean)
    shoulder_y = hip_y - torso_length_px

    place("nose", 0, shoulder_y - 130)
    place("leftEye", 22, shoulder_y - 145)
    place("rightEye", -22, shoulder_y - 145)
    place("leftEyeInner", 12, shoulder_y - 145)
    place("rightEyeInner", -12, shoulder_y - 145)
    place("leftEyeOuter", 32, shoulder_y - 145)
    place("rightEyeOuter", -32, shoulder_y - 145)
    place("leftEar", 46, shoulder_y - 138)
    place("rightEar", -46, shoulder_y - 138)
    place("leftMouth", 16, shoulder_y - 100)
    place("rightMouth", -16, shoulder_y - 100)

    place("leftShoulder", 80, shoulder_y)
    place("rightShoulder", -80, shoulder_y)
    place("leftHip", 55 + hip_back, hip_y)
    place("rightHip", -55 + hip_back, hip_y)

    place("leftKnee", 55 + knee_forward, knee_y)
    place("rightKnee", -55 + knee_forward, knee_y)
    place("leftAnkle", 55, ankle_y)
    place("rightAnkle", -55, ankle_y)
    place("leftHeel", 50, ankle_y + 16)
    place("rightHeel", -50, ankle_y + 16)
    place("leftFootIndex", 85, ankle_y + 12)
    place("rightFootIndex", -85, ankle_y + 12)

    angle = math.radians(90 - arm_raise * 90)
    arm = 110.0
    for side, sign in (("left", 1), ("right", -1)):
        sx = 80 * sign
        ex = sx + sign * arm * math.cos(angle) * 0.55
        ey = shoulder_y + arm * math.sin(angle) * 0.55
        place(f"{side}Elbow", ex, ey)
        wx = ex + sign * arm * math.cos(angle) * 0.75
        wy = ey + arm * math.sin(angle) * 0.75
        place(f"{side}Wrist", wx, wy)
        place(f"{side}Thumb", wx + sign * 10, wy + 12)
        place(f"{side}Index", wx + sign * 16, wy + 16)
        place(f"{side}Pinky", wx + sign * 6, wy + 18)

    return pts, vis


def knee_angle(pts: np.ndarray) -> float:
    """Interior knee angle from a synthetic pose."""
    h = pts[IDX["leftHip"]]
    k = pts[IDX["leftKnee"]]
    a = pts[IDX["leftAnkle"]]
    v1, v2 = h - k, a - k
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


_failures: list[str] = []
_passes = 0


def check(condition: bool, message: str) -> None:
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(message)

def test_dead_zone_is_enforced() -> None:
    """A configuration inside the measured noise band must be rejected.

    The published limits of agreement on knee angle are -6.7 to +11.9 degrees.
    A dead zone narrower than that is not a tuning choice, it is a defect, and
    the constructor refuses it rather than letting it ship.
    """
    try:
        m.HysteresisConfig("bad", down_below=100, up_above=110)
        check(False, "a 10 degree dead zone should have been rejected")
    except ValueError:
        check(True, "")

    try:
        m.HysteresisConfig("inverted", down_below=160, up_above=100)
        check(False, "reversed thresholds should have been rejected")
    except ValueError:
        check(True, "")


def test_counts_a_clean_squat() -> None:
    counter = m.RepCounter(m.SQUAT)
    t = 0.0
    step = 1 / 30

    def feed(bend: float, frames: int) -> list[m.RepResult]:
        """Returns every frame's result.

        The completion flag is raised on the single frame the repetition
        finishes, not on every frame afterwards - so a test that only inspects
        the last frame of a phase will miss it. An earlier version of this test
        did exactly that and reported a failure that was not there.
        """
        nonlocal t
        results = []
        for _ in range(frames):
            pts, _ = make_pose(knee_bend=bend)
            results.append(counter.update(knee_angle(pts), t))
            t += step
        return results

    feed(0.0, 20)                       # standing
    check(counter.state is m.RepState.UP, "should start in UP while standing")

    feed(0.95, 20)                      # descend and hold
    check(counter.state is m.RepState.DOWN, "should reach DOWN in a deep squat")

    results = feed(0.0, 20)             # stand back up
    check(counter.count == 1, f"expected 1 rep, got {counter.count}")
    check(any(r.completed for r in results),
          "no frame reported the repetition completing")


def test_jitter_cannot_produce_a_phantom_rep() -> None:
    """The central claim: a wide dead zone makes angle noise harmless.

    Published noise is +/-12 degrees. Here the angle is jittered by +/-15,
    worse than measured, right at the standing position. A narrow threshold
    would chatter; a 60 degree dead zone cannot.
    """
    counter = m.RepCounter(m.SQUAT)
    t = 0.0
    pts, _ = make_pose(knee_bend=0.0)
    base = knee_angle(pts)

    for i in range(200):
        noisy = base + (15.0 if i % 2 else -15.0)
        counter.update(noisy, t)
        t += 1 / 30

    check(counter.count == 0,
          f"jitter produced {counter.count} phantom reps")


def test_a_single_frame_spike_is_ignored() -> None:
    """Confirmation over consecutive frames kills spikes hysteresis cannot.

    A spike large enough to cross a 60 degree dead zone still lasts one frame.
    Limb swaps and detection glitches look exactly like this.
    """
    counter = m.RepCounter(m.SQUAT)
    t = 0.0
    for i in range(120):
        angle = 30.0 if i == 60 else 175.0   # one frame claiming a deep squat
        counter.update(angle, t)
        t += 1 / 30

    check(counter.count == 0, "a one-frame spike was counted as a rep")
    check(counter.state is m.RepState.UP, "state should have stayed UP")


def test_partial_rep_is_reported_not_ignored() -> None:
    """Going partway down and back up is useful feedback, not silence."""
    counter = m.RepCounter(m.SQUAT)
    t = 0.0
    step = 1 / 30

    def feed(angle: float, frames: int) -> m.RepResult:
        nonlocal t
        result = None
        for _ in range(frames):
            result = counter.update(angle, t)
            t += step
        return result

    feed(175.0, 20)                     # standing
    feed(130.0, 20)                     # into the dead zone, never reaching DOWN
    result = feed(175.0, 20)            # back up

    check(result.partial_count == 1,
          f"expected 1 partial, got {result.partial_count}")
    check(result.count == 0, "a partial must not count as a full rep")


def test_minimum_dwell_rejects_impossible_speed() -> None:
    """A squat that bottoms out for a few milliseconds did not happen.

    Note what has to be held constant for this to test what it claims. An
    earlier version fed the fast frames at 300 fps and then returned to 30 fps.
    But the return to UP needs three confirmation frames, and at 30 fps those
    take 100 ms - so by the time the transition was accepted, the bottom *had*
    been held for over 300 ms and the counter was right to count it.

    Here the whole sequence runs at 300 fps, so the bottom is genuinely brief.
    """
    counter = m.RepCounter(m.SQUAT)
    t = 0.0
    fast = 1 / 300

    for _ in range(60):
        counter.update(175.0, t)
        t += fast
    for _ in range(6):
        counter.update(60.0, t)
        t += fast
    for _ in range(60):
        counter.update(175.0, t)
        t += fast

    check(counter.count == 0,
          f"a {6 * fast * 1000:.0f} ms bottom was counted as a rep")
    check(counter.partial_count == 0,
          "a single fast movement should not be reported")


def test_repeated_bouncing_is_reported() -> None:
    """Bouncing is the pattern worth surfacing, not one fast movement.

    This is the other half of the dwell-time behaviour: silence on a single
    borderline movement, feedback once it becomes a habit.
    """
    counter = m.RepCounter(m.SQUAT)
    t = 0.0
    fast = 1 / 300

    for _ in range(60):
        counter.update(175.0, t)
        t += fast
    for _ in range(3):
        for _ in range(6):
            counter.update(60.0, t)
            t += fast
        for _ in range(6):
            counter.update(175.0, t)
            t += fast
    for _ in range(30):
        counter.update(175.0, t)
        t += fast

    check(counter.count == 0, "bouncing was counted as real repetitions")
    check(counter.partial_count == 1,
          f"repeated bouncing should be reported once, got "
          f"{counter.partial_count}")


def test_clean_reps_produce_no_false_partials() -> None:
    """Found by driving realistic movement rather than step changes.

    The dwell counter originally incremented on every frame a transition was
    held back - but waiting out the dwell time is exactly what a normal
    repetition does. At 30 fps a 300 ms dwell is nine frames, so every clean
    squat accumulated nine 'blocks' and was reported as a partial on its way to
    being counted as a success. Five ideal squats produced four phantom
    partials.

    Perfect movement must produce zero complaints.
    """
    session = m.MovementSession()
    t = 0.0
    for _ in range(5):
        for i in range(36):
            bend = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2 * 0.95
            pts, vis = make_pose(knee_bend=bend)
            session.update(pts, vis, knee_angle(pts), t)
            t += 1 / 30

    check(session.rep_counter.count == 5,
          f"expected 5 clean squats, counted {session.rep_counter.count}")
    check(session.rep_counter.partial_count == 0,
          f"perfect squats produced {session.rep_counter.partial_count} "
          "phantom partials")


def test_missing_angle_is_not_a_state() -> None:
    """None means 'not measured', which must not be read as movement."""
    counter = m.RepCounter(m.SQUAT)
    t = 0.0
    for _ in range(20):
        counter.update(175.0, t)
        t += 1 / 30
    before = counter.state
    for _ in range(30):
        counter.update(None, t)
        t += 1 / 30
    check(counter.state is before, "an occlusion changed the state")
    check(counter.count == 0, "an occlusion produced a rep")


def test_arm_raise_inverted() -> None:
    """The same machinery, with the working position at the top."""
    counter = m.RepCounter(m.ARM_RAISE, inverted=True)
    t = 0.0
    step = 1 / 30
    for _ in range(20):
        counter.update(20.0, t)     # arms down
        t += step
    for _ in range(20):
        counter.update(165.0, t)    # arms up
        t += step
    result = None
    for _ in range(20):
        result = counter.update(20.0, t)
        t += step
    check(result.count == 1, f"expected 1 arm raise, got {result.count}")

def test_com_sits_inside_the_body() -> None:
    pts, vis = make_pose()
    com = m.centre_of_mass(pts, vis)
    check(com is not None, "CoM should be computable from a full pose")
    hip_y = (pts[IDX["leftHip"]][1] + pts[IDX["rightHip"]][1]) / 2
    sh_y = (pts[IDX["leftShoulder"]][1] + pts[IDX["rightShoulder"]][1]) / 2
    check(sh_y < com[1] < pts[IDX["leftKnee"]][1],
          f"CoM at y={com[1]:.0f} is outside the trunk")
    check(abs(com[0] - 360.0) < 20,
          f"a symmetric pose should give a centred CoM, got x={com[0]:.0f}")


def test_com_refuses_when_the_body_is_mostly_unseen() -> None:
    """A CoM from three landmarks is a different quantity, not a worse one."""
    pts, vis = make_pose()
    for name in ("leftHip", "rightHip", "leftKnee", "rightKnee",
                 "leftAnkle", "rightAnkle", "leftShoulder", "rightShoulder"):
        vis[IDX[name]] = 0.1
    check(m.centre_of_mass(pts, vis) is None,
          "CoM was computed from too little of the body")


def test_com_is_steadier_than_a_single_joint() -> None:
    """The structural claim behind using CoM for jumps.

    Independent per-landmark noise cancels in a mass-weighted average. This is
    why published CoM tracking reaches 21 mm RMSE while single joints carry
    several times that.

    Tested directly: identical noise applied to every landmark, then compare the
    scatter of the CoM against the scatter of one joint.
    """
    rng = np.random.default_rng(42)
    base, vis = make_pose()

    com_samples = []
    wrist_samples = []
    for _ in range(400):
        noisy = base + rng.normal(0, 3.0, base.shape)
        com = m.centre_of_mass(noisy, vis)
        if com is not None:
            com_samples.append(com)
            wrist_samples.append(noisy[IDX["leftWrist"]])

    com_arr = np.array(com_samples)
    wrist_arr = np.array(wrist_samples)
    com_std = float(np.mean([com_arr[:, 0].std(), com_arr[:, 1].std()]))
    wrist_std = float(np.mean([wrist_arr[:, 0].std(), wrist_arr[:, 1].std()]))

    check(com_std < wrist_std * 0.5,
          f"CoM std {com_std:.2f} should be far below a joint's {wrist_std:.2f}")

def simulate_jump(detector: m.JumpDetector, peak_metres: float = 0.30,
                  fps: int = 60, standing_seconds: float = 1.0,
                  tiptoe: bool = False) -> tuple[m.JumpEvent | None, float]:
    """Drive the detector with a physically correct jump, or a tiptoe rise.

    A real jump follows y = v*t - g*t^2/2 during flight. Tiptoes rises smoothly
    and stops - no free fall - which is exactly the distinction the detector's
    parabola fit is meant to catch.
    """
    step = 1.0 / fps
    t = 0.0
    torso_px = 200.0
    torso_metres = 0.5
    px_per_metre = torso_px / torso_metres
    ground_y = 600.0
    event = None

    for _ in range(int(standing_seconds * fps)):
        com = np.array([360.0, ground_y])
        e = detector.update(com, torso_px, t)
        event = event or e
        t += step

    if tiptoe:
        n = int(0.5 * fps)
        for i in range(n):
            h = peak_metres * math.sin(math.pi * i / n)
            com = np.array([360.0, ground_y - h * px_per_metre])
            e = detector.update(com, torso_px, t)
            event = event or e
            t += step
    else:
        v0 = math.sqrt(2 * m.GRAVITY * peak_metres)
        flight = 2 * v0 / m.GRAVITY
        for i in range(int(flight * fps) + 1):
            dt = i * step
            h = max(0.0, v0 * dt - 0.5 * m.GRAVITY * dt * dt)
            com = np.array([360.0, ground_y - h * px_per_metre])
            e = detector.update(com, torso_px, t)
            event = event or e
            t += step

    for _ in range(int(0.5 * fps)):
        com = np.array([360.0, ground_y])
        e = detector.update(com, torso_px, t)
        event = event or e
        t += step

    return event, t


def test_detects_a_real_jump() -> None:
    detector = m.JumpDetector()
    event, _ = simulate_jump(detector, peak_metres=0.30)
    check(event is not None, "a 30 cm jump was not detected")
    if event:
        expected_flight = 2 * math.sqrt(2 * m.GRAVITY * 0.30) / m.GRAVITY
        check(abs(event.flight_seconds - expected_flight) < 0.12,
              f"flight {event.flight_seconds:.3f}s vs expected "
              f"{expected_flight:.3f}s")
        check(abs(event.fitted_gravity_ratio - 1.0) < 0.5,
              f"fitted gravity ratio {event.fitted_gravity_ratio:.2f} "
              "should be near 1 for a real jump")


def test_rejects_standing_on_tiptoes() -> None:
    """The discrimination the literature does not cover, decided by physics.

    Tiptoes rises the same distance but has no free-fall phase, so the fitted
    acceleration is near zero rather than near g.
    """
    detector = m.JumpDetector()
    event, _ = simulate_jump(detector, peak_metres=0.12, tiptoe=True)
    check(event is None, "rising onto tiptoes was reported as a jump")


def test_ignores_postural_sway() -> None:
    detector = m.JumpDetector()
    rng = np.random.default_rng(7)
    t = 0.0
    for _ in range(300):
        com = np.array([360.0 + rng.normal(0, 2), 600.0 + rng.normal(0, 3)])
        event = detector.update(com, 200.0, t)
        check(event is None, "swaying was reported as a jump")
        t += 1 / 60


def test_losing_the_subject_aborts_the_jump() -> None:
    """A jump that cannot be observed to its end must not be reported."""
    detector = m.JumpDetector()
    t = 0.0
    for _ in range(60):
        detector.update(np.array([360.0, 600.0]), 200.0, t)
        t += 1 / 60
    for i in range(10):
        detector.update(np.array([360.0, 600.0 - i * 8.0]), 200.0, t)
        t += 1 / 60
    for _ in range(10):
        event = detector.update(None, None, t)
        check(event is None, "a jump was reported after tracking was lost")
        t += 1 / 60

def test_median_filter_rejects_a_spike() -> None:
    filt = m.MedianFilter(5)
    base, _ = make_pose()
    t = 0.0
    out = None
    for i in range(12):
        pts = base.copy()
        if i == 8:
            pts[IDX["leftWrist"]] += 400.0     # a limb-swap sized jump
        out = filt(pts, t)
        t += 1 / 30
    check(np.allclose(out[IDX["leftWrist"]], base[IDX["leftWrist"]], atol=1.0),
          "the median filter let a spike through")


def test_median_filter_resets_across_a_gap() -> None:
    """ML Kit's 100 ms rule: frames either side of a gap are not adjacent."""
    filt = m.MedianFilter(5)
    base, _ = make_pose()
    for i in range(10):
        filt(base, i / 30)
    moved = base + 50.0
    out = filt(moved, 10 / 30 + 0.5)      # a 500 ms gap
    check(np.allclose(out, moved),
          "state from before a long gap was blended into the new frame")


def test_viewpoint_gate_detects_turning_away() -> None:
    pts, vis = make_pose()
    facing = m.viewpoint_offset_degrees(pts, vis)
    check(facing is not None and facing < 35.0,
          f"a front-facing pose read as {facing} degrees off")

    turned = pts.copy()
    turned[IDX["nose"]][0] = 360.0 + 62
    turned[IDX["leftShoulder"]][0] = 360.0 + 70
    turned[IDX["rightShoulder"]][0] = 360.0 - 20
    offset = m.viewpoint_offset_degrees(turned, vis)
    check(offset is not None and offset > 35.0,
          f"a side-on pose read as only {offset} degrees off")


def test_3d_yaw_beats_the_2d_estimate() -> None:
    """The defect this pins, found against ground truth on a real photograph.

    The 2D nose-based estimate reported 64 degrees of yaw for a subject whose
    true yaw - from MediaPipe's 3D world landmarks - was 6.7 degrees. The gate
    refused to analyse someone facing the camera.

    The cause is anatomical: the nose follows the *head*, and a person can turn
    their head while their torso squarely faces the camera. Glancing aside is
    not a reason to stop measuring their squat.
    """
    square = np.zeros((33, 3))
    square[IDX["leftShoulder"]] = (0.2, 0.0, 0.0)
    square[IDX["rightShoulder"]] = (-0.2, 0.0, 0.0)
    yaw = m.torso_yaw_degrees(square)
    check(yaw is not None and yaw < 5.0,
          f"a square-on torso measured {yaw} degrees of yaw")

    turned = np.zeros((33, 3))
    turned[IDX["leftShoulder"]] = (0.2, 0.0, 0.2)
    turned[IDX["rightShoulder"]] = (-0.2, 0.0, -0.2)
    yaw = m.torso_yaw_degrees(turned)
    check(yaw is not None and 40 < yaw < 50,
          f"a 45-degree torso measured {yaw} degrees")

    profile = np.zeros((33, 3))
    profile[IDX["leftShoulder"]] = (0.0, 0.0, 0.2)
    profile[IDX["rightShoulder"]] = (0.0, 0.0, -0.2)
    yaw = m.torso_yaw_degrees(profile)
    check(yaw is not None and yaw > 80.0,
          f"a full profile measured {yaw} degrees")


def test_session_prefers_3d_yaw_when_available() -> None:
    """A head turned aside must not stop analysis when the torso is square."""
    pts, vis = make_pose()
    pts = pts.copy()
    pts[IDX["nose"]][0] = 360.0 + 62

    twod = m.viewpoint_offset_degrees(pts, vis)
    check(twod is not None and twod > 35.0,
          "fixture is wrong: the 2D estimate should object here")

    square = np.zeros((33, 3))
    square[IDX["leftShoulder"]] = (0.2, 0.0, 0.0)
    square[IDX["rightShoulder"]] = (-0.2, 0.0, 0.0)

    session = m.MovementSession()
    report = session.update(pts, vis, 175.0, 0.0, world_points=square)
    check(report.usable,
          f"3D yaw was available and square, but the frame was refused: "
          f"{report.reason}")


def test_session_still_refuses_a_genuinely_turned_torso() -> None:
    """The gate must not become permissive - only accurate."""
    pts, vis = make_pose()
    profile = np.zeros((33, 3))
    profile[IDX["leftShoulder"]] = (0.05, 0.0, 0.2)
    profile[IDX["rightShoulder"]] = (-0.05, 0.0, -0.2)

    session = m.MovementSession()
    report = session.update(pts, vis, 175.0, 0.0, world_points=profile)
    check(not report.usable, "a genuinely side-on torso was analysed")
    check("face the camera" in report.reason,
          f"unhelpful reason: {report.reason!r}")


def test_shoulder_ratio_tracks_yaw() -> None:
    pts, vis = make_pose()
    front = m.shoulder_torso_ratio(pts, vis)
    turned = pts.copy()
    turned[IDX["leftShoulder"]][0] = 360.0 + 10
    turned[IDX["rightShoulder"]][0] = 360.0 - 10
    profile = m.shoulder_torso_ratio(turned, vis)
    check(front is not None and profile is not None and profile < front * 0.5,
          f"shoulder ratio {front} -> {profile} should collapse in profile")


def test_subject_switch_on_a_size_jump() -> None:
    detector = m.SubjectSwitchDetector()
    pts, vis = make_pose(scale=1.0)
    t = 0.0
    for _ in range(10):
        check(not detector(pts, vis, t), "a steady subject triggered a switch")
        t += 1 / 30
    taller, vis2 = make_pose(scale=1.45)
    check(detector(taller, vis2, t),
          "a 45 percent torso change was not detected as a subject switch")


def test_subject_switch_detects_a_modest_build_change() -> None:
    """The guard must catch more than a caricature.

    The old torso threshold of 7.0 per second was above every build change
    under 23%, so in practice the hip test was doing all the work - and it was
    the hip test that fired on jumps. A 15% difference in build is an ordinary
    difference between two adults and gives 4.50 /s, which must be caught.
    """
    for scale in (0.85, 1.15):
        detector = m.SubjectSwitchDetector()
        pts, vis = make_pose(scale=1.0)
        t = 0.0
        for _ in range(10):
            detector(pts, vis, t)
            t += 1 / 30
        other, other_vis = make_pose(scale=scale)
        check(detector(other, other_vis, t),
              f"a {abs(1 - scale) * 100:.0f} percent build change was missed")


def test_subject_switch_survives_an_explosive_jump() -> None:
    """The measured defect, as a test.

    A countermovement jump with a 133 ms push-off reaches a hip rate of about
    14 torso-lengths per second. Measured over 399 physiologically-timed
    movements, nothing real exceeds 15.22 /s across two consecutive frames.
    """
    detector = m.SubjectSwitchDetector()
    fps = 30.0
    drop = 0.45          # metres of hip drop at full knee bend
    pixels_per_metre = 400.0

    def frame(bend: float, airborne_m: float):
        pts, vis = make_pose(knee_bend=bend)
        pts = pts.copy()
        pts[:, 1] -= (airborne_m - bend * drop) * pixels_per_metre
        return pts, vis

    def ease(a: float, b: float, n: int) -> list[float]:
        return [a + (b - a) * (1 - math.cos(math.pi * (i + 1) / n)) / 2
                for i in range(n)]

    sequence: list[tuple[float, float]] = []
    sequence += [(0.0, 0.0)] * 8
    sequence += [(b, 0.0) for b in ease(0.0, 0.55, 10)]   # countermovement
    sequence += [(b, 0.0) for b in ease(0.55, 0.0, 4)]    # 133 ms push-off
    take_off = (2 * 9.81 * 0.40) ** 0.5
    flight = int(round((2 * take_off / 9.81) * fps))
    for i in range(flight):
        s = (i + 1) / fps
        sequence.append((0.0, max(0.0, take_off * s - 0.5 * 9.81 * s * s)))
    sequence += [(b, 0.0) for b in ease(0.0, 0.38, 3)]    # 100 ms absorption
    sequence += [(b, 0.0) for b in ease(0.38, 0.0, 10)]
    sequence += [(0.0, 0.0)] * 8

    fired = []
    for i, (bend, air) in enumerate(sequence):
        pts, vis = frame(bend, air)
        if detector(pts, vis, i / fps):
            fired.append(i)

    check(not fired,
          f"an explosive 40 cm jump was reported as a subject switch "
          f"at frames {fired}")


def test_scale_invariance_of_the_rep_counter() -> None:
    """The same movement at two distances must count the same.

    Angles are scale-free by construction, so this verifies the pose fixture
    and the angle computation agree with that.
    """
    for scale in (0.6, 1.0, 1.6):
        counter = m.RepCounter(m.SQUAT)
        t = 0.0
        for bend, frames in ((0.0, 20), (0.95, 20), (0.0, 20)):
            for _ in range(frames):
                pts, _ = make_pose(knee_bend=bend, scale=scale)
                counter.update(knee_angle(pts), t)
                t += 1 / 30
        check(counter.count == 1,
              f"at scale {scale} the count was {counter.count}, expected 1")

def test_session_declines_when_turned_away() -> None:
    session = m.MovementSession()
    pts, vis = make_pose()
    turned = pts.copy()
    turned[IDX["nose"]][0] = 360.0 + 62
    turned[IDX["leftShoulder"]][0] = 360.0 + 70
    turned[IDX["rightShoulder"]][0] = 360.0 - 20
    report = session.update(turned, vis, 175.0, 0.0)
    check(not report.usable, "the session analysed a side-on subject")
    check("face the camera" in report.reason,
          f"unhelpful reason: {report.reason!r}")


def squat_sequence(session: m.MovementSession, start_t: float = 0.0,
                   fps: int = 30) -> float:
    """Drive a session through one anatomically plausible squat.

    The descent and ascent are **ramped over several frames** rather than
    jumping from standing to bottom in one. That matters: a body cannot fold
    instantaneously, and the subject-switch detector correctly treats an
    instantaneous change as a different person. An earlier version of this test
    stepped straight from bend 0.0 to 0.95 and then blamed the detector for
    firing - the fixture was unphysical, not the code.
    """
    t = start_t
    step = 1 / fps

    def ramp(a: float, b: float, frames: int) -> None:
        nonlocal t
        for i in range(frames):
            bend = a + (b - a) * (i + 1) / frames
            pts, vis = make_pose(knee_bend=bend)
            session.update(pts, vis, knee_angle(pts), t)
            t += step

    def hold(bend: float, frames: int) -> None:
        nonlocal t
        for _ in range(frames):
            pts, vis = make_pose(knee_bend=bend)
            session.update(pts, vis, knee_angle(pts), t)
            t += step

    hold(0.0, 20)       # standing
    ramp(0.0, 0.95, 12)  # descend over 400 ms
    hold(0.95, 15)      # hold the bottom
    ramp(0.95, 0.0, 12)  # rise
    hold(0.0, 20)       # standing again
    return t


def test_session_counts_a_squat_end_to_end() -> None:
    session = m.MovementSession()
    squat_sequence(session)
    check(session.rep_counter.count == 1,
          f"end to end count was {session.rep_counter.count}")


def test_session_keeps_counts_across_a_subject_switch() -> None:
    """A switch invalidates the rep in progress, not the ones already done."""
    session = m.MovementSession()
    t = squat_sequence(session)
    check(session.rep_counter.count == 1, "setup should have counted one rep")

    taller, vis2 = make_pose(scale=1.5)
    report = session.update(taller, vis2, 175.0, t + 1 / 30)
    check(report.subject_switched, "the switch was not detected")
    check(report.rep_count == 1,
          f"completed reps were lost on a switch: {report.rep_count}")


def test_a_real_squat_does_not_trigger_a_subject_switch() -> None:
    """The defect this pins.

    A squat changes the apparent torso length by over 100% as the body folds
    and foreshortens. A magnitude threshold fires on every repetition; a rate
    threshold does not, because the change is gradual.
    """
    session = m.MovementSession()
    t = 0.0
    step = 1 / 30
    switches = 0
    for i in range(24):
        bend = (math.sin(2 * math.pi * i / 24 - math.pi / 2) + 1) / 2 * 0.95
        pts, vis = make_pose(knee_bend=bend)
        report = session.update(pts, vis, knee_angle(pts), t)
        switches += int(report.subject_switched)
        t += step
    check(switches == 0,
          f"a normal squat triggered {switches} false subject switches")

def test_non_squat_movements_count_nothing() -> None:
    """Everyday movement must not be read as exercise.

    False positives are worse than misses here. A missed repetition is a
    nuisance; a phantom one means the app is reporting movement the user did
    not make, which is the failure mode that destroys trust in the whole
    signal.

    Each of these is a plausible thing to do in front of a camera, and none of
    them is a squat.
    """
    rng = np.random.default_rng(1)

    def run(frames) -> m.MovementSession:
        session = m.MovementSession()
        t = 0.0
        for pts, vis, angle in frames:
            session.update(pts, vis, angle, t)
            t += 1 / 30
        return session

    def noisy(**kwargs):
        pts, vis = make_pose(**kwargs)
        pts = pts + rng.normal(0, 3, pts.shape)
        return pts, vis, knee_angle(pts)

    still = [noisy() for _ in range(300)]
    walking = [noisy(knee_bend=0.25 + 0.12 * math.sin(2 * math.pi * i / 20))
               for i in range(300)]
    sitting = [noisy(knee_bend=min(0.95, i / 40) if i < 120 else 0.95)
               for i in range(300)]
    waving = [noisy(arm_raise=(math.sin(2 * math.pi * i / 25) + 1) / 2)
              for i in range(300)]

    for label, frames in (("standing still", still),
                          ("walking in place", walking),
                          ("sitting down once", sitting),
                          ("waving arms", waving)):
        session = run(frames)
        check(session.rep_counter.count == 0,
              f"{label} produced {session.rep_counter.count} phantom squats")
        check(session.jump_count == 0,
              f"{label} produced {session.jump_count} phantom jumps")


def test_survives_dropped_frames() -> None:
    """Measured: counting is unaffected up to 25% frame loss.

    Beyond 40% it degrades - 4.1 of 5 - which is the honest limit rather than a
    cliff. Worth knowing, because a mid-range phone under thermal load drops
    frames rather than slowing down smoothly.
    """
    rng = np.random.default_rng(0)
    session = m.MovementSession()
    t = 0.0
    for _ in range(5):
        for i in range(36):
            bend = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2 * 0.95
            if rng.random() < 0.25:
                t += 1 / 30
                continue
            pts, vis = make_pose(knee_bend=bend)
            pts = pts + rng.normal(0, 3, pts.shape)
            session.update(pts, vis, knee_angle(pts), t)
            t += 1 / 30
    check(session.rep_counter.count == 5,
          f"25% frame loss gave {session.rep_counter.count} of 5 squats")


def test_intermittent_occlusion_is_harmless() -> None:
    """Measured: 5/5 squats and zero false partials under 15% occlusion.

    This works because a missing angle is passed as None and treated as absence
    of information rather than as a state. A leg hidden behind furniture for a
    few frames is the normal case in a bedroom, not an edge case.
    """
    rng = np.random.default_rng(0)
    session = m.MovementSession()
    t = 0.0
    for _ in range(5):
        for i in range(36):
            bend = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2 * 0.95
            pts, vis = make_pose(knee_bend=bend)
            pts = pts + rng.normal(0, 3, pts.shape)
            angle = knee_angle(pts)
            if rng.random() < 0.15:
                vis = vis.copy()
                vis[IDX["leftKnee"]] = 0.1
                vis[IDX["leftAnkle"]] = 0.1
                angle = None
            session.update(pts, vis, angle, t)
            t += 1 / 30
    check(session.rep_counter.count == 5,
          f"occlusion gave {session.rep_counter.count} of 5 squats")
    check(session.rep_counter.partial_count == 0,
          f"occlusion produced {session.rep_counter.partial_count} false "
          "partials")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            _failures.append(f"{test.__name__} raised {exc!r}")

    print(f"\n  {_passes} checks passed, {len(_failures)} failed")
    for failure in _failures:
        print(f"    FAIL: {failure}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
