from __future__ import annotations

import numpy as np

QUIET_THRESHOLD = 0.005

RELEASE_THRESHOLD = 0.015


class JointAnchor:
    """Holds each joint still while it is quiet, releases it when it moves.

    Stateful per landmark, and safe to call with a changing set of landmarks:
    anything not previously seen is adopted at its current position.
    """

    def __init__(self, quiet: float = QUIET_THRESHOLD,
                 release: float = RELEASE_THRESHOLD) -> None:
        if release <= quiet:
            raise ValueError("release threshold must exceed the quiet one")
        self.quiet = quiet
        self.release = release
        self._held: np.ndarray | None = None

    def update(self, points: np.ndarray, torso: float,
               usable: np.ndarray | None = None,
               frame_size: tuple[int, int] | None = None) -> np.ndarray:
        """Anchor this frame's joints. Returns the positions to use.

        `frame_size` is `(width, height)`. It exists because the anchor runs
        *after* the out-of-frame check, and holding a joint near the boundary
        can place it a pixel or two outside - measured, 10 landmarks across 28
        clips ended up 0.9 to 2.2 px past an edge, having already been passed
        as drawable. Given the frame size, held positions are kept inside it.
        """
        if torso <= 1e-6 or not np.isfinite(torso):
            return points
        if self._held is None or len(self._held) != len(points):
            self._held = points.copy().astype(np.float64)
            return points

        if not np.all(np.isfinite(self._held)):
            self._held = points.copy().astype(np.float64)
            return points
        step = np.linalg.norm(points - self._held, axis=1) / torso
        blend = np.clip((step - self.quiet) / (self.release - self.quiet),
                        0.0, 1.0)
        held = self._held + (points - self._held) * blend[:, None]

        if usable is not None:
            held[~usable] = points[~usable]

        bad = ~np.all(np.isfinite(held), axis=1)
        if bad.any():
            held[bad] = points[bad]

        if frame_size is not None:
            w, h = frame_size
            inside = np.all(
                (points >= [-w * 0.002, -h * 0.002])
                & (points <= [w * 1.002, h * 1.002]), axis=1)
            if inside.any():
                held[inside, 0] = np.clip(held[inside, 0], 0.0, float(w))
                held[inside, 1] = np.clip(held[inside, 1], 0.0, float(h))

        self._held = held
        return held.copy()

    def reset(self) -> None:
        self._held = None