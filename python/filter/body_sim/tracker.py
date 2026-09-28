from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TARGET_SUBJECT_HEIGHT = 500

MAX_UPSCALE = 8.0

SEARCH_ATTEMPTS: tuple[tuple[float, float], ...] = (
    (0.7, 1.0),
    (0.7, 0.55),
    (0.7, 0.3),
    (2.0, 1.0),
)

MAX_PREDICTION_SECONDS = 0.5

EAGER_SEARCH_SECONDS = 1.0

SEARCH_INTERVAL_SECONDS = 0.25

MAX_SEARCH_INTERVAL_SECONDS = 1.0

GIVE_UP_AFTER_SECONDS = None


@dataclass
class SearchRegion:
    """A box to look for the subject in, and how to map results back."""

    x0: int
    y0: int
    x1: int
    y1: int
    scale: float

    @property
    def valid(self) -> bool:
        return self.x1 - self.x0 >= 16 and self.y1 - self.y0 >= 16

    def to_frame(self, points: np.ndarray, crop_w: int,
                 crop_h: int) -> np.ndarray:
        """Map landmarks detected inside the crop back to frame pixels."""
        out = points.copy().astype(np.float64)
        out[:, 0] = self.x0 + out[:, 0] / self.scale
        out[:, 1] = self.y0 + out[:, 1] / self.scale
        return out


class SubjectLocator:
    """Remembers where the subject was, and where they are heading.

    Deliberately tiny: two positions and two timestamps. A heavier motion model
    would be guessing at detail this cannot support - the point is only to aim
    the search region, not to produce a measurement.
    """

    def __init__(self) -> None:
        self._centre: np.ndarray | None = None
        self._size: float | None = None
        self._velocity = np.zeros(2, dtype=np.float64)
        self._time: float | None = None
        self._last_search: float | None = None
        self._interval = SEARCH_INTERVAL_SECONDS

    @property
    def has_subject(self) -> bool:
        return self._centre is not None and self._size is not None

    def seen(self, points: np.ndarray, visibility: np.ndarray,
             timestamp: float) -> None:
        """Record a frame in which the subject was found."""
        good = points[visibility >= 0.5]
        if len(good) < 4:
            return
        low = good.min(axis=0)
        high = good.max(axis=0)
        centre = (low + high) / 2
        size = float(max(high[0] - low[0], high[1] - low[1]))
        if not np.all(np.isfinite(centre)) or not np.isfinite(size) or size <= 0:
            return

        if self._centre is not None and self._time is not None:
            dt = timestamp - self._time
            if 1e-3 < dt < 0.5:
                instant = (centre - self._centre) / dt
                self._velocity = 0.5 * self._velocity + 0.5 * instant

        self._centre = centre
        self._size = size
        self._time = timestamp
        self._last_search = None
        self._interval = SEARCH_INTERVAL_SECONDS

    def should_search(self, timestamp: float) -> bool:
        """Whether it is worth looking for the subject on this frame.

        Searching costs 12.5 ms when it finds nothing, so doing it on every
        frame of an empty room would halve the frame rate for no purpose. The
        first second of absence is searched eagerly - that is when a brief
        occlusion or a fast movement is recoverable - then it drops to a poll,
        then stops.
        """
        if not self.has_subject or self._time is None:
            return False
        missing = timestamp - self._time
        if GIVE_UP_AFTER_SECONDS is not None and missing > GIVE_UP_AFTER_SECONDS:
            return False
        if missing <= EAGER_SEARCH_SECONDS:
            return True
        if self._last_search is None:
            return True
        return timestamp - self._last_search >= self._interval

    def searched(self, timestamp: float) -> None:
        """Record a failed search, and back off.

        Exponential, because the longer someone has been gone the less likely
        they are to reappear in the next quarter second - and every look costs a
        full inference.

        The back-off only starts *after* the eager window. Doubling during it
        was measured to break recovery entirely: by the time the subject
        returned, the interval had already reached its maximum, so nothing
        looked for them. That is the same stranding the whole module exists to
        prevent, reintroduced by its own cost control.
        """
        self._last_search = timestamp
        if self._time is None or timestamp - self._time <= EAGER_SEARCH_SECONDS:
            self._interval = SEARCH_INTERVAL_SECONDS
            return
        self._interval = min(self._interval * 2, MAX_SEARCH_INTERVAL_SECONDS)

    def reset(self) -> None:
        self._centre = None
        self._size = None
        self._velocity = np.zeros(2, dtype=np.float64)
        self._time = None
        self._last_search = None
        self._interval = SEARCH_INTERVAL_SECONDS

    def attempts(self) -> int:
        """How many distinct searches are available."""
        return len(SEARCH_ATTEMPTS)

    def region(self, attempt: int, frame_w: int, frame_h: int,
               timestamp: float) -> SearchRegion | None:
        """Where to look, on this attempt. None ends the search.

        Attempts vary two things: how wide a region to cover, and how big the
        subject is assumed to be. Size is varied because the subject being
        invisible to the full-frame detector usually means they are *smaller*
        than when last seen, and a region sized for their old size gives an
        upscale of 1.0 - which is no search at all.
        """
        if not self.has_subject or attempt >= len(SEARCH_ATTEMPTS):
            return None

        padding, size_factor = SEARCH_ATTEMPTS[attempt]
        assumed_size = self._size * size_factor

        elapsed = 0.0 if self._time is None else max(0.0, timestamp - self._time)
        horizon = min(elapsed, MAX_PREDICTION_SECONDS)
        centre = self._centre + self._velocity * horizon

        half = assumed_size * (0.5 + padding)
        x0 = int(max(0, centre[0] - half))
        y0 = int(max(0, centre[1] - half))
        x1 = int(min(frame_w, centre[0] + half))
        y1 = int(min(frame_h, centre[1] + half))

        width = x1 - x0
        height = y1 - y0
        if width < 16 or height < 16:
            return None

        scale = TARGET_SUBJECT_HEIGHT / max(assumed_size, 1.0)
        scale = float(min(max(scale, 1.0), MAX_UPSCALE))

        if (width >= frame_w * 0.9 and height >= frame_h * 0.9
                and scale <= 1.01):
            return None

        return SearchRegion(x0, y0, x1, y1, scale)


def crop_for_search(image: np.ndarray,
                    region: SearchRegion) -> np.ndarray | None:
    """Cut and upscale the search region, ready for detection."""
    if not region.valid:
        return None
    crop = image[region.y0:region.y1, region.x0:region.x1]
    if crop.size == 0:
        return None
    if region.scale > 1.0:
        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * region.scale)),
             max(1, int(crop.shape[0] * region.scale))),
            interpolation=cv2.INTER_CUBIC)
    return crop