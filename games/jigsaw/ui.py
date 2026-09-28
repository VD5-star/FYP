from __future__ import annotations
from dataclasses import dataclass, field
TOUCH_MIN = 44
@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int
    def hit(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h
    @property
    def cx(self) -> int:
        return self.x + self.w // 2
    @property
    def cy(self) -> int:
        return self.y + self.h // 2
    @property
    def x1(self) -> int:
        return self.x + self.w
    @property
    def y1(self) -> int:
        return self.y + self.h
    def touchable(self) -> bool:
        return self.w >= TOUCH_MIN and self.h >= TOUCH_MIN
@dataclass
class Zones:
    items: dict[str, Rect] = field(default_factory=dict)
    def add(self, key: str, r: Rect) -> Rect:
        self.items[key] = r
        return r
    def at(self, px: int, py: int) -> str | None:
        for k, r in reversed(list(self.items.items())):
            if r.hit(px, py):
                return k
        return None
    def clear(self) -> None:
        self.items.clear()
    def __contains__(self, key: str) -> bool:
        return key in self.items
    def __getitem__(self, key: str) -> Rect:
        return self.items[key]
