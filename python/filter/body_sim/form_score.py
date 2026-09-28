from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def resample(sequence: list[float], length: int = 32) -> np.ndarray:
    """Resamples a trajectory to a fixed length.

    Repetitions differ in duration, and comparing raw sequences would confuse
    "performed differently" with "performed slower". Resampling separates the
    two: shape is compared here, and duration is reported separately.
    """
    if not sequence:
        return np.zeros(length)
    data = np.asarray(sequence, dtype=np.float64)
    if len(data) == 1:
        return np.full(length, data[0])
    source = np.linspace(0.0, 1.0, len(data))
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, data)


def dtw_distance(a: np.ndarray, b: np.ndarray,
                 band: int | None = None) -> float:
    """Dynamic time warping distance between two trajectories.


    Two correct repetitions can be correctly shaped but differently paced -
    someone may pause at the bottom on one and not the next. A pointwise
    comparison counts that as a large error; DTW aligns them in time first, so
    it measures *shape* difference rather than *timing* difference.


    A Sakoe-Chiba band limits how far the alignment may stray from the
    diagonal. Without it, DTW can match the start of one repetition to the end
    of another and report a suspiciously good score for two unlike movements.
    The band both prevents that and reduces the cost.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float('inf')
    if band is None:
        band = max(4, max(n, m) // 4)

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j],
                                 cost[i, j - 1],
                                 cost[i - 1, j - 1])

    result = cost[n, m]
    if not np.isfinite(result):
        return float('inf')
    return float(result / (n + m))


@dataclass
class RepetitionRecord:
    """One completed repetition, with enough to compare it against others."""

    angles: list[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def depth(self) -> float | None:
        """The extreme reached, as the minimum angle."""
        return min(self.angles) if self.angles else None

    @property
    def range_of_motion(self) -> float | None:
        if not self.angles:
            return None
        return max(self.angles) - min(self.angles)


@dataclass
class FormReport:
    """How a repetition compared with the reference."""

    similarity: float
    """0 to 1, where 1 is identical in shape to the reference."""

    depth_difference: float | None = None
    """Degrees shallower (positive) or deeper (negative) than the reference."""

    tempo_ratio: float | None = None
    """Duration relative to the reference. Above 1 is slower."""

    notes: list[str] = field(default_factory=list)
    """Plain descriptions of what differed, for showing to the user."""


class FormAnalyser:
    """Compares each completed repetition against a reference.


    By default the **first completed repetition of the session** becomes the
    reference. That is deliberate: comparing a person against themselves avoids
    every problem with comparing them against a population - body proportions,
    mobility, injury history, camera placement - none of which this engine can
    see or should be judging.

    A supplied reference can override it, for a guided programme.
    """

    NOTABLE_DIFFERENCE = 0.25

    NOTABLE_DEPTH = 20.0

    def __init__(self, reference: RepetitionRecord | None = None,
                 resample_length: int = 32) -> None:
        self.reference = reference
        self.resample_length = resample_length
        self.history: list[RepetitionRecord] = []
        self._reference_shape: np.ndarray | None = None
        self._skipped_first = reference is not None
        if reference is not None:
            self._reference_shape = resample(reference.angles, resample_length)

    DISCARD_FIRST = True

    def add(self, record: RepetitionRecord) -> FormReport | None:
        """Records a completed repetition and scores it.

        Returns None until a reference exists, because inventing a score with
        nothing to compare against would be fabricating information.
        """
        if not record.angles:
            return None

        self.history.append(record)

        if self.reference is None:
            if self.DISCARD_FIRST and not self._skipped_first:
                self._skipped_first = True
                return None
            self.reference = record
            self._reference_shape = resample(record.angles,
                                             self.resample_length)
            return None

        shape = resample(record.angles, self.resample_length)
        assert self._reference_shape is not None
        distance = dtw_distance(shape, self._reference_shape)

        scale = self.reference.range_of_motion or 1.0
        similarity = float(np.clip(1.0 - distance / max(scale, 1e-6), 0.0, 1.0))

        notes: list[str] = []

        depth_difference = None
        if record.depth is not None and self.reference.depth is not None:
            depth_difference = record.depth - self.reference.depth
            if depth_difference > self.NOTABLE_DEPTH:
                notes.append('shallower than your first repetition')
            elif depth_difference < -self.NOTABLE_DEPTH:
                notes.append('deeper than your first repetition')

        tempo_ratio = None
        if self.reference.duration > 1e-6:
            tempo_ratio = record.duration / self.reference.duration
            if tempo_ratio > 1.5:
                notes.append('slower than your first repetition')
            elif tempo_ratio < 0.67:
                notes.append('faster than your first repetition')

        if similarity < (1.0 - self.NOTABLE_DIFFERENCE) and not notes:
            notes.append('moved differently from your first repetition')

        return FormReport(
            similarity=similarity,
            depth_difference=depth_difference,
            tempo_ratio=tempo_ratio,
            notes=notes,
        )

    @property
    def consistency(self) -> float | None:
        """How alike the session's repetitions were, 0 to 1.

        More useful than any single score: consistency is visible to the user
        as a meaningful property of their set, whereas one repetition's
        similarity is mostly noise.
        """
        if len(self.history) < 2:
            return None

        scored = self.history[1:] if self.DISCARD_FIRST else self.history
        shapes = [resample(r.angles, self.resample_length) for r in scored]
        reference = self._reference_shape
        if reference is None:
            return None

        scale = max(self.reference.range_of_motion or 1.0, 1e-6)
        scores = [
            float(np.clip(1.0 - dtw_distance(s, reference) / scale, 0.0, 1.0))
            for s in shapes[1:]
        ]
        return float(np.mean(scores)) if scores else None

    def reset(self) -> None:
        self.reference = None
        self._reference_shape = None
        self._skipped_first = False
        self.history.clear()


class RepetitionCollector:
    """Accumulates the angle trajectory between repetition boundaries.

    Sits alongside a counter: every frame's angle is offered here, and when the
    counter reports a completed repetition the accumulated trajectory is closed
    and handed to the analyser.
    """

    def __init__(self) -> None:
        self._angles: list[float] = []
        self._start: float | None = None

    def add(self, angle: float | None, timestamp: float) -> None:
        if angle is None:
            return
        if self._start is None:
            self._start = timestamp
        self._angles.append(angle)

    def close(self, timestamp: float) -> RepetitionRecord | None:
        """Ends the current repetition and returns it.

        Returns None when too few frames were collected to describe a movement -
        a three-frame "repetition" is not something to score.
        """
        if len(self._angles) < 8 or self._start is None:
            self._angles = []
            self._start = None
            return None

        record = RepetitionRecord(
            angles=list(self._angles),
            start_time=self._start,
            end_time=timestamp,
        )
        self._angles = []
        self._start = timestamp
        return record

    def reset(self) -> None:
        self._angles = []
        self._start = None