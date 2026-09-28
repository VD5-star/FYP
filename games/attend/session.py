from __future__ import annotations
import random
from dataclasses import dataclass, field
import sounds
FOCUS = "focus"
SWITCH = "switch"
DIVIDE = "divide"
SETTLE = "settle"
OVER = "over"
MIN_MINUTES = 3
MAX_MINUTES = 10
DEFAULT_MINUTES = 5
SETTLE_FRAC = 0.06
SETTLE_MIN = 10.0
SETTLE_MAX = 26.0
DIVIDE_FRAC = 0.28
FOCUS_FRAC = 0.34
SWITCH_HOLD = 6.0
MIN_SWITCHES = 6
CROSSFADE = 1.1
SHAPES = (
    (4, "light", "a short pass, one look at each sound"),
    (6, "steady", "long enough to settle into it"),
    (8, "full", "every sound gets real time"),
    (10, "deep", "a long hold at the end"),
)
def clamp_minutes(m: int) -> int:
    return int(min(MAX_MINUTES, max(MIN_MINUTES, int(m))))
def shape(minutes: int) -> tuple[str, str]:
    m = clamp_minutes(minutes)
    for top, name, detail in SHAPES:
        if m <= top:
            return name, detail
    return SHAPES[-1][1], SHAPES[-1][2]
def clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"
@dataclass
class Step:
    stage: str
    target: str | None
    hold: float
@dataclass
class Session:
    minutes: int = DEFAULT_MINUTES
    seed: int | None = None
    steps: list[Step] = field(default_factory=list)
    index: int = 0
    elapsed: float = 0.0
    paused: bool = False
    finished: bool = False
    def __post_init__(self):
        self.build()
    def build(self) -> None:
        self.minutes = clamp_minutes(self.minutes)
        total = float(self.minutes * 60)
        rng = random.Random(self.seed)
        names = list(sounds.NAMES)
        settle = min(SETTLE_MAX, max(SETTLE_MIN, total * SETTLE_FRAC))
        divide = total * DIVIDE_FRAC
        focus_all = total * FOCUS_FRAC
        focus_hold = focus_all / len(names)
        moving = total - settle - divide - focus_all
        switches = max(MIN_SWITCHES, int(round(moving / SWITCH_HOLD)))
        switch_hold = moving / switches
        steps: list[Step] = []
        order = names[:]
        rng.shuffle(order)
        for n in order:
            steps.append(Step(FOCUS, n, focus_hold))
        last = steps[-1].target if steps else None
        for _ in range(switches):
            choices = [n for n in names if n != last]
            n = rng.choice(choices)
            steps.append(Step(SWITCH, n, switch_hold))
            last = n
        steps.append(Step(DIVIDE, None, divide))
        steps.append(Step(SETTLE, None, settle))
        drift = total - sum(s.hold for s in steps)
        steps[-1].hold = max(1.0, steps[-1].hold + drift)
        self.steps = steps
        self.index = 0
        self.elapsed = 0.0
        self.finished = False
        self.paused = False
    @property
    def total(self) -> float:
        return sum(s.hold for s in self.steps)
    @property
    def step(self) -> Step | None:
        if self.finished or self.index >= len(self.steps):
            return None
        return self.steps[self.index]
    @property
    def stage(self) -> str:
        s = self.step
        return OVER if s is None else s.stage
    @property
    def target(self) -> str | None:
        s = self.step
        return None if s is None else s.target
    @property
    def done_before(self) -> float:
        return sum(s.hold for s in self.steps[:self.index])
    @property
    def position(self) -> float:
        return self.done_before + self.elapsed
    @property
    def progress(self) -> float:
        t = self.total
        return 0.0 if t <= 0 else min(1.0, self.position / t)
    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.position)
    @property
    def step_progress(self) -> float:
        s = self.step
        if s is None or s.hold <= 0:
            return 1.0
        return min(1.0, self.elapsed / s.hold)
    def gains(self) -> dict[str, float]:
        s = self.step
        if s is None:
            return {n: 0.0 for n in sounds.NAMES}
        return {n: 1.0 for n in sounds.NAMES}
    def scene_weights(self) -> dict[str, float]:
        s = self.step
        out = {n: 0.0 for n in sounds.NAMES}
        if s is None:
            return out
        if s.stage in (DIVIDE, SETTLE):
            for n in out:
                out[n] = 1.0 / len(out)
            return out
        prev = None
        for k in range(self.index - 1, -1, -1):
            if self.steps[k].target:
                prev = self.steps[k].target
                break
        t = min(1.0, self.elapsed / CROSSFADE) if CROSSFADE > 0 else 1.0
        if prev and prev != s.target and t < 1.0:
            out[prev] = 1.0 - t
            out[s.target] = t
        else:
            out[s.target] = 1.0
        return out
    MAX_STEP = 0.25
    def advance(self, dt: float, clamp: bool = True) -> bool:
        if self.finished or self.paused or dt <= 0:
            return False
        moved = False
        if clamp:
            dt = min(dt, self.MAX_STEP)
        while dt > 0 and not self.finished:
            s = self.step
            if s is None:
                self.finished = True
                break
            left = s.hold - self.elapsed
            if dt < left:
                self.elapsed += dt
                dt = 0.0
            else:
                dt -= left
                self.index += 1
                self.elapsed = 0.0
                moved = True
                if self.index >= len(self.steps):
                    self.finished = True
        return moved
    def toggle_pause(self) -> bool:
        if not self.finished:
            self.paused = not self.paused
        return self.paused
    def restart(self) -> None:
        self.build()
