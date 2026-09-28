from __future__ import annotations
import threading
import numpy as np
import sounds
SR = sounds.SR
BLOCK = 1024
VOL_MIN = 0
VOL_MAX = 100
VOL_DEFAULT = 55
MASTER = 0.72
RAMP = 0.06
FADE_SECONDS = 1.4
def _curve(v: int) -> float:
    if v <= VOL_MIN:
        return 0.0
    f = float(np.clip(v, VOL_MIN, VOL_MAX)) / VOL_MAX
    return float(f ** 2.2)
class Audio:
    def __init__(self, volume: int = VOL_DEFAULT, sr: int = SR,
                 block: int = BLOCK, seconds: float | None = None) -> None:
        self.sr = sr
        self.block = block
        self.volume = int(np.clip(volume, VOL_MIN, VOL_MAX))
        self.lock = threading.Lock()
        self.stream = None
        self.failed = False
        self.pos = 0
        self.gain = 0.0
        self.target = _curve(self.volume)
        self.fade = 0.0
        self.fade_step = block / (FADE_SECONDS * sr)
        self.clipped = 0
        self.underflow = 0
        self.bank: dict = {}
        self.order: list[str] = []
        self.mix: np.ndarray | None = None
        self.ready = threading.Event()
        self.seconds = seconds or sounds.LOOP_SECONDS
        self._t = threading.Thread(target=self._build, daemon=True)
        self._t.start()
    def _build(self) -> None:
        bank = sounds.render_all(self.seconds, self.sr)
        order = list(sounds.NAMES)
        n = len(bank[order[0]])
        mix = np.zeros((n, 2), np.float32)
        for name in order:
            pan = sounds.PANS.get(name, 0.0)
            lg = float(np.sqrt(0.5 * (1.0 - pan)))
            rg = float(np.sqrt(0.5 * (1.0 + pan)))
            x = bank[name]
            mix[:, 0] += x * lg * 1.41
            mix[:, 1] += x * rg * 1.41
        peak = float(np.abs(mix).max())
        if peak > 0.98:
            mix *= 0.98 / peak
        with self.lock:
            self.bank = bank
            self.order = order
            self.mix = mix
        self.ready.set()
    def wait(self, timeout: float | None = None) -> bool:
        return self.ready.wait(timeout)
    @property
    def level_name(self) -> str:
        return "off" if self.volume <= VOL_MIN else str(self.volume)
    @property
    def on(self) -> bool:
        return self.stream is not None and not self.failed
    def start(self) -> bool:
        if self.stream is not None or self.failed:
            return self.on
        try:
            import sounddevice as sd
            self.stream = sd.OutputStream(
                samplerate=self.sr, channels=2, dtype="float32",
                blocksize=self.block, callback=self._callback)
            self.stream.start()
        except Exception:
            self.stream = None
            self.failed = True
            return False
        return True
    def stop(self) -> None:
        s = self.stream
        self.stream = None
        with self.lock:
            self.pos = 0
            self.gain = 0.0
            self.fade = 0.0
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
    def set_volume(self, v: int) -> int:
        with self.lock:
            self.volume = int(np.clip(v, VOL_MIN, VOL_MAX))
            self.target = _curve(self.volume)
        return self.volume
    def nudge(self, d: int) -> int:
        return self.set_volume(self.volume + d)
    def render(self, frames: int) -> np.ndarray:
        with self.lock:
            mix = self.mix
            target = self.target
        if mix is None:
            self.pos += frames
            return np.zeros((frames, 2), np.float32)
        n = len(mix)
        i = self.pos % n
        if i + frames <= n:
            out = mix[i:i + frames].copy()
        else:
            head = mix[i:]
            out = np.concatenate([head, mix[:frames - len(head)]])
        out = out * MASTER
        if self.fade < 1.0:
            self.fade = min(1.0, self.fade + self.fade_step)
        g1 = target * self.fade
        ramp = np.linspace(self.gain, g1, frames,
                           dtype=np.float32)[:, None]
        out = out * ramp
        self.gain = g1
        peak = float(np.abs(out).max())
        if peak > 1.0:
            self.clipped += 1
            out = out / peak
        self.pos += frames
        return out.astype(np.float32)
    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self.underflow += 1
        try:
            outdata[:] = self.render(frames)
        except Exception:
            outdata.fill(0.0)
