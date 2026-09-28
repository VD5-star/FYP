from __future__ import annotations

import sys

import numpy as np

import pose_sim
from test_movement import make_pose

_failures: list[str] = []
_passes = 0


def check(condition: bool, message: str) -> None:
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(message)


def test_working_size_shrinks_to_target() -> None:
    check(pose_sim.working_size(2560, 1440, 720) == (1280, 720),
          'a 16:9 frame should shrink to exactly the target height')
    check(pose_sim.working_size(1000, 666, 480) == (721, 480),
          'a non-standard aspect should keep its ratio when shrunk')
    check(pose_sim.working_size(720, 1280, 720) == (405, 720),
          'a portrait frame should shrink by height like any other')


def test_working_size_never_enlarges() -> None:
    """Enlarging would invent pixels and cost time to process them."""
    check(pose_sim.working_size(640, 480, 720) is None,
          'a frame already shorter than the target must be left alone')
    check(pose_sim.working_size(1280, 720, 720) is None,
          'a frame exactly at the target needs no resize')


def test_working_size_zero_disables() -> None:
    """--height 0 and --full-size must mean "do not touch the frame"."""
    check(pose_sim.working_size(2560, 1440, 0) is None,
          'a target of 0 should disable resizing')
    check(pose_sim.working_size(2560, 1440, -10) is None,
          'a negative target should disable resizing, not invert the frame')


def test_working_size_stays_positive() -> None:
    """A very wide, very short frame must not round its width down to zero."""
    size = pose_sim.working_size(4000, 20, 2)
    check(size is not None and size[0] >= 1 and size[1] >= 1,
          f'degenerate frame produced a non-positive size: {size}')


def test_fit_to_screen_fits() -> None:
    """The whole point: the window must be no larger than the screen."""
    screen_w, screen_h = pose_sim.screen_size()
    for width, height in ((2560, 1440), (3840, 2160), (720, 1280), (1000, 666)):
        win_w, win_h = pose_sim.fit_to_screen(width, height)
        check(win_w <= screen_w and win_h <= screen_h,
              f'{width}x{height} gave a {win_w}x{win_h} window on a '
              f'{screen_w}x{screen_h} screen')


def test_fit_to_screen_limits_by_height_too() -> None:
    """Fitting width alone is the classic mistake.

    A 2560x1440 frame scaled to a 1920-wide screen is 1080 tall, which is
    taller than the work area once the taskbar is removed - so a width-only fit
    puts the feet behind the taskbar, on a tool whose whole job is watching
    feet.
    """
    screen_w, screen_h = pose_sim.screen_size()
    win_w, win_h = pose_sim.fit_to_screen(400, 4000)
    check(win_h <= screen_h,
          f'a very tall frame gave a {win_h} px window on a {screen_h} px screen')
    check(win_w <= screen_w, 'width should still be within the screen')


def test_fit_to_screen_keeps_aspect() -> None:
    """A stretched preview would misrepresent every angle drawn on it."""
    for width, height in ((2560, 1440), (720, 1280), (1000, 666), (640, 480)):
        win_w, win_h = pose_sim.fit_to_screen(width, height)
        source = width / height
        shown = win_w / win_h
        check(abs(source - shown) / source < 0.02,
              f'{width}x{height} was shown at aspect {shown:.3f} '
              f'instead of {source:.3f}')


def test_fit_to_screen_never_enlarges() -> None:
    """A small frame blown up to fill the screen would look sharp and lie."""
    win_w, win_h = pose_sim.fit_to_screen(320, 240)
    check((win_w, win_h) == (320, 240),
          f'a small frame was enlarged to {win_w}x{win_h}')


def test_screen_size_is_usable() -> None:
    width, height = pose_sim.screen_size()
    check(width > 0 and height > 0,
          f'screen size must be positive, got {width}x{height}')
    check(width >= 640 and height >= 480,
          f'screen size {width}x{height} is implausibly small; the fallback '
          f'should have applied')


def test_default_height_is_within_measured_range() -> None:
    """The default must stay inside the range accuracy was measured over.

    Angle agreement with full-resolution detection was measured from 1080 down
    to 360. A default outside that window would be an untested claim.
    """
    check(360 <= pose_sim.DEFAULT_WORKING_HEIGHT <= 1080,
          f'default height {pose_sim.DEFAULT_WORKING_HEIGHT} is outside the '
          f'360-1080 range that was actually measured')


def _inked_fraction(canvas, a, b, samples: int = 60, radius: int = 3) -> float:
    """Fraction of points along the segment a-b that have ink within `radius`.

    A tolerance is needed because anti-aliasing and the dark casing mean the
    exact centre pixel of a line is not reliably the bright colour.
    """
    hits = 0
    for i in range(samples):
        p = a + (b - a) * ((i + 0.5) / samples)
        x, y = int(round(p[0])), int(round(p[1]))
        y0, y1 = max(0, y - radius), min(canvas.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(canvas.shape[1], x + radius + 1)
        if canvas[y0:y1, x0:x1].any():
            hits += 1
    return hits / samples


def _trunk_canvas():
    pts, vis = make_pose(knee_bend=0.0)
    canvas = np.zeros((1000, 720, 3), np.uint8)
    pose_sim.draw_trunk(canvas, pts, vis, 7.2)
    return pts, vis, canvas


def test_trunk_draws_a_spine() -> None:
    """The axis every measurement uses should be visible.

    The torso frame's origin is mid-hip and its unit vector runs to
    mid-shoulder. Drawing that line makes the coordinate system visible, so a
    viewer can see when it is wrong.
    """
    pts, _, canvas = _trunk_canvas()
    mid_sh = (pts[pose_sim.IDX['leftShoulder']]
              + pts[pose_sim.IDX['rightShoulder']]) / 2
    mid_hip = (pts[pose_sim.IDX['leftHip']]
               + pts[pose_sim.IDX['rightHip']]) / 2
    check(_inked_fraction(canvas, mid_hip, mid_sh) > 0.95,
          'no spine was drawn between mid-hip and mid-shoulder')


def test_trunk_draws_both_cross_bars() -> None:
    pts, _, canvas = _trunk_canvas()
    shoulder = _inked_fraction(canvas, pts[pose_sim.IDX['leftShoulder']],
                               pts[pose_sim.IDX['rightShoulder']])
    hip = _inked_fraction(canvas, pts[pose_sim.IDX['leftHip']],
                          pts[pose_sim.IDX['rightHip']])
    check(shoulder > 0.95, f'shoulder bar only {shoulder:.0%} drawn')
    check(hip > 0.95, f'hip bar only {hip:.0%} drawn')


def test_no_quadrilateral_across_the_chest() -> None:
    """There is no shoulder-to-hip bone in a skeleton.

    The trunk used to be drawn as four bones forming a box, which reads as a
    crate rather than a body. The sides must be clear - checked at their
    midpoints, which are far from both the spine and the cross-bars.
    """
    pts, _, canvas = _trunk_canvas()
    for side, top, bottom in (
        ('left', 'leftShoulder', 'leftHip'),
        ('right', 'rightShoulder', 'rightHip'),
    ):
        mid = (pts[pose_sim.IDX[top]] + pts[pose_sim.IDX[bottom]]) / 2
        x, y = int(round(mid[0])), int(round(mid[1]))
        patch = canvas[y - 6:y + 7, x - 6:x + 7]
        check(not patch.any(),
              f'the {side} side of the old chest box is still drawn')


def test_no_trunk_bones_remain_in_the_bone_list() -> None:
    """The source of the box, not just its appearance."""
    trunk = {pose_sim.IDX[n] for n in
             ('leftShoulder', 'rightShoulder', 'leftHip', 'rightHip')}
    spanning = [(a, b) for a, b, _ in pose_sim.BONES if {a, b} <= trunk]
    check(not spanning,
          f'bones spanning only trunk landmarks remain: {spanning}')


def test_shoulders_and_hips_keep_their_joint_dots() -> None:
    """They are real joints, and the limbs hinge at them."""
    for name in ('leftShoulder', 'rightShoulder', 'leftHip', 'rightHip'):
        check(pose_sim.IDX[name] in pose_sim.MAJOR_JOINTS,
              f'{name} lost its joint dot')


def test_trunk_refuses_when_a_landmark_is_not_visible() -> None:
    """Drawing a spine from a guessed hip would be inventing the body's axis."""
    pts, vis, _ = _trunk_canvas()
    hidden = vis.copy()
    hidden[pose_sim.IDX['leftHip']] = 0.05
    blank = np.zeros((1000, 720, 3), np.uint8)
    drew = pose_sim.draw_trunk(blank, pts, hidden, 7.2)
    check(not drew, 'the trunk drew with an invisible hip')
    check(not blank.any(), 'the trunk drew pixels while reporting refusal')


def test_skeleton_still_renders_end_to_end() -> None:
    """The whole figure, not just the part that changed."""
    for bend, arms in ((0.0, 0.0), (0.85, 0.0), (0.0, 1.0)):
        pts, vis = make_pose(knee_bend=bend, arm_raise=arms)
        canvas = np.full((1000, 720, 3), 30, np.uint8)
        pose_sim.draw_skeleton(canvas, pts, vis, 'full')
        coverage = (canvas != 30).any(axis=2).mean()
        check(coverage > 0.01,
              f'skeleton at bend {bend} arms {arms} drew almost nothing '
              f'({coverage:.2%})')


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            _failures.append(f'{test.__name__} raised {exc!r}')

    print(f'\n  {_passes} checks passed, {len(_failures)} failed')
    for failure in _failures:
        print(f'    FAIL: {failure}')
    return 1 if _failures else 0


if __name__ == '__main__':
    sys.exit(main())