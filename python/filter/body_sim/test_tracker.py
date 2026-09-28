from __future__ import annotations

import sys

import numpy as np

import tracker as tr

_failures: list[str] = []
_passes = 0


def check(condition: bool, message: str) -> None:
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(message)


FRAME_W, FRAME_H = 1280, 720


def subject(centre_x: float = 640.0, centre_y: float = 360.0,
            height: float = 400.0):
    """Landmarks spanning a box of the given height, all confident."""
    half_h = height / 2
    half_w = height / 4
    points = np.array([
        [centre_x - half_w, centre_y - half_h],
        [centre_x + half_w, centre_y - half_h],
        [centre_x - half_w, centre_y + half_h],
        [centre_x + half_w, centre_y + half_h],
        [centre_x, centre_y],
    ], dtype=np.float64)
    return points, np.ones(len(points))


def test_starts_with_no_subject() -> None:
    locator = tr.SubjectLocator()
    check(not locator.has_subject, 'a fresh locator claimed to know the subject')
    check(locator.region(0, FRAME_W, FRAME_H, 0.0) is None,
          'a fresh locator offered a search region')
    check(not locator.should_search(0.0),
          'a fresh locator wanted to search for a subject it never saw')


def test_remembers_where_the_subject_was() -> None:
    locator = tr.SubjectLocator()
    points, visibility = subject(centre_x=300.0, centre_y=200.0, height=240.0)
    locator.seen(points, visibility, 0.0)
    check(locator.has_subject, 'the locator did not record the subject')

    region = locator.region(0, FRAME_W, FRAME_H, 0.1)
    check(region is not None, 'no search region after seeing the subject')
    if region is None:
        return
    check(region.x0 < 300 < region.x1 and region.y0 < 200 < region.y1,
          f'the search region {region} does not contain where the subject was')


def test_unreliable_landmarks_are_ignored() -> None:
    """A box drawn around guessed landmarks would search the wrong place."""
    locator = tr.SubjectLocator()
    points, visibility = subject()
    blind = visibility.copy()
    blind[:] = 0.1
    locator.seen(points, blind, 0.0)
    check(not locator.has_subject,
          'the locator recorded a position from landmarks it could not trust')


def test_the_search_actually_magnifies() -> None:
    """The defect that made the whole feature useless.

    Sizing the crop to a fixed output resolution gave a 785x693 region a scale
    of 1.00 - no magnification at all - so a subject who left at 480 px and
    returned at 140 px was never found, because the search was doing nothing.

    The scale must come from the subject's expected height, not the region's.
    """
    locator = tr.SubjectLocator()
    points, visibility = subject(height=400.0)
    locator.seen(points, visibility, 0.0)
    for attempt in range(locator.attempts()):
        region = locator.region(attempt, FRAME_W, FRAME_H, 0.1)
        if region is None:
            continue
        check(region.scale > 1.0,
              f'search attempt {attempt} had scale {region.scale:.2f} - it '
              f'would hand the detector the same pixels that already failed')


def test_later_attempts_look_for_a_smaller_subject() -> None:
    """Being invisible to the detector usually means having moved away."""
    locator = tr.SubjectLocator()
    points, visibility = subject(height=400.0)
    locator.seen(points, visibility, 0.0)
    scales = []
    for attempt in range(locator.attempts()):
        region = locator.region(attempt, FRAME_W, FRAME_H, 0.1)
        if region is not None:
            scales.append(region.scale)
    check(max(scales) > min(scales) * 1.5,
          f'the attempts barely differ in magnification: {scales} - a subject '
          f'who moved further away would never be found')


def test_the_search_ends() -> None:
    """An unbounded search would be an infinite loop in the caller."""
    locator = tr.SubjectLocator()
    points, visibility = subject()
    locator.seen(points, visibility, 0.0)
    check(locator.region(locator.attempts(), FRAME_W, FRAME_H, 0.1) is None,
          'asking past the last attempt still returned a region')


def test_prediction_leads_a_moving_subject() -> None:
    """MediaPipe centres its crop on where the subject *was*.

    That is the documented design - there is no motion model anywhere in its
    graph - and it is why a fast movement escapes the 1.25x crop margin.
    """
    locator = tr.SubjectLocator()
    for i in range(4):
        points, visibility = subject(centre_x=300.0 + i * 60)
        locator.seen(points, visibility, i * 0.1)

    still = tr.SubjectLocator()
    points, visibility = subject(centre_x=480.0)
    still.seen(points, visibility, 0.3)

    moving_region = locator.region(0, FRAME_W, FRAME_H, 0.6)
    still_region = still.region(0, FRAME_W, FRAME_H, 0.6)
    if moving_region is None or still_region is None:
        check(False, 'no region produced for the prediction test')
        return
    moving_centre = (moving_region.x0 + moving_region.x1) / 2
    still_centre = (still_region.x0 + still_region.x1) / 2
    check(moving_centre > still_centre + 20,
          f'the search region did not lead the moving subject: '
          f'{moving_centre:.0f} vs {still_centre:.0f}')


def test_prediction_is_bounded() -> None:
    """Extrapolation diverges; past a short horizon the wider search is better."""
    locator = tr.SubjectLocator()
    for i in range(4):
        points, visibility = subject(centre_x=300.0 + i * 60)
        locator.seen(points, visibility, i * 0.1)

    last_seen = 0.3
    beyond = last_seen + tr.MAX_PREDICTION_SECONDS + 0.2
    near = locator.region(0, FRAME_W, FRAME_H, beyond)
    far = locator.region(0, FRAME_W, FRAME_H, 30.0)
    if near is None or far is None:
        check(False, 'no region produced for the prediction bound test')
        return
    check(abs((far.x0 + far.x1) / 2 - (near.x0 + near.x1) / 2) < 1.0,
          'prediction kept extrapolating after 30 seconds, which would put '
          'the search region somewhere the subject certainly is not')

    unpredicted = tr.SubjectLocator()
    points, visibility = subject(centre_x=480.0)
    unpredicted.seen(points, visibility, last_seen)
    plain = unpredicted.region(0, FRAME_W, FRAME_H, beyond)
    if plain is not None:
        check((far.x0 + far.x1) / 2 > (plain.x0 + plain.x1) / 2 + 20,
              'the capped prediction is indistinguishable from none at all')


def test_searching_is_eager_at_first() -> None:
    """The first second of absence is the recoverable case."""
    locator = tr.SubjectLocator()
    points, visibility = subject()
    locator.seen(points, visibility, 0.0)
    for t in (0.03, 0.2, 0.5, 0.9):
        check(locator.should_search(t),
              f'the locator would not search {t} s after losing the subject')
        locator.searched(t)


def test_searching_backs_off_but_never_stops() -> None:
    """A fixed give-up was measured to strand a returning user.

    The last known position stays valid indefinitely - the subject left a room
    whose camera did not move - so the region is still the right place to look
    much later.
    """
    locator = tr.SubjectLocator()
    points, visibility = subject()
    locator.seen(points, visibility, 0.0)

    searches = 0
    t = 0.0
    while t < 60.0:
        if locator.should_search(t):
            locator.searched(t)
            searches += 1
        t += 1 / 30

    check(searches > 0, 'the locator gave up searching entirely')
    check(searches < 200,
          f'{searches} searches in a minute of absence - the back-off is not '
          f'working and an empty room would cost a fortune')
    check(locator.should_search(3600.0) is not None,
          'searching stopped after an hour')


def test_back_off_does_not_start_during_the_eager_window() -> None:
    """A defect found by measurement.

    Doubling the interval during the eager window meant that by the time the
    subject returned, the interval had already reached its maximum - so nothing
    looked for them. The cost control reintroduced the exact stranding the
    module exists to prevent.
    """
    locator = tr.SubjectLocator()
    points, visibility = subject()
    locator.seen(points, visibility, 0.0)

    t = 0.0
    while t <= tr.EAGER_SEARCH_SECONDS:
        if locator.should_search(t):
            locator.searched(t)
        t += 1 / 30

    check(locator._interval <= tr.SEARCH_INTERVAL_SECONDS + 1e-9,
          f'the interval grew to {locator._interval} s during the eager '
          f'window, so a returning subject would not be looked for')


def test_finding_the_subject_resets_the_back_off() -> None:
    locator = tr.SubjectLocator()
    points, visibility = subject()
    locator.seen(points, visibility, 0.0)
    for t in (2.0, 4.0, 8.0, 16.0):
        if locator.should_search(t):
            locator.searched(t)
    check(locator._interval > tr.SEARCH_INTERVAL_SECONDS,
          'the interval never grew, so the back-off is not being exercised')

    locator.seen(points, visibility, 20.0)
    check(locator._interval == tr.SEARCH_INTERVAL_SECONDS,
          'the back-off survived finding the subject again')


def test_regions_stay_inside_the_frame() -> None:
    """A crop reaching outside the image would fail or wrap."""
    locator = tr.SubjectLocator()
    for x, y in ((20.0, 20.0), (1260.0, 700.0), (640.0, 10.0)):
        locator.reset()
        points, visibility = subject(centre_x=x, centre_y=y, height=300.0)
        locator.seen(points, visibility, 0.0)
        for attempt in range(locator.attempts()):
            region = locator.region(attempt, FRAME_W, FRAME_H, 0.1)
            if region is None:
                continue
            check(0 <= region.x0 < region.x1 <= FRAME_W
                  and 0 <= region.y0 < region.y1 <= FRAME_H,
                  f'region {region} escapes the frame for a subject at ({x},{y})')


def test_mapping_back_to_the_frame() -> None:
    """Landmarks found in the crop must land where the subject really is."""
    region = tr.SearchRegion(x0=200, y0=100, x1=456, y1=356, scale=2.0)
    local = np.array([[256.0, 256.0], [0.0, 0.0]])
    mapped = region.to_frame(local, 512, 512)
    check(abs(mapped[0][0] - 328.0) < 1e-9 and abs(mapped[0][1] - 228.0) < 1e-9,
          f'crop centre mapped to {mapped[0]}, expected (328, 228)')
    check(abs(mapped[1][0] - 200.0) < 1e-9 and abs(mapped[1][1] - 100.0) < 1e-9,
          f'crop origin mapped to {mapped[1]}, expected (200, 100)')


def test_crop_refuses_a_degenerate_region() -> None:
    image = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    tiny = tr.SearchRegion(x0=10, y0=10, x1=12, y1=12, scale=4.0)
    check(tr.crop_for_search(image, tiny) is None,
          'a two-pixel region was cropped and handed to the detector')


def test_crop_is_magnified() -> None:
    image = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    region = tr.SearchRegion(x0=100, y0=100, x1=200, y1=200, scale=3.0)
    crop = tr.crop_for_search(image, region)
    check(crop is not None and crop.shape[0] == 300 and crop.shape[1] == 300,
          f'crop came out {None if crop is None else crop.shape}, '
          f'expected 300x300')


def test_reset_forgets_everything() -> None:
    locator = tr.SubjectLocator()
    points, visibility = subject()
    locator.seen(points, visibility, 0.0)
    locator.reset()
    check(not locator.has_subject, 'reset left the subject behind')
    check(not locator.should_search(1.0), 'reset left the search running')


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