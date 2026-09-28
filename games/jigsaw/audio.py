from __future__ import annotations
import threading
import numpy as np
SR = 44100
BLOCK = 512
VOL_MIN = 0
VOL_MAX = 100
VOL_DEFAULT = 55
def _curve(v: int) -> float:
    if v <= VOL_MIN:
        return 0.0
    f = float(np.clip(v, VOL_MIN, VOL_MAX)) / VOL_MAX
    return float(f ** 2.2)
CLICK_GAIN = 0.34
CLICK_MS = 150.0
DONE_GAIN = 0.30
MAX_VOICES = 8
TONES = (196.0, 220.0, 247.0, 262.0, 294.0, 330.0)
DONE_TONE = 147.0
ATTACK_MS = 9.0
def _click(sr: int = SR, ms: float = CLICK_MS, seed: int = 0,
           freq: float = 247.0) -> np.ndarray:
    n = max(16, int(sr * ms / 1000.0))
    t = np.arange(n, dtype=np.float32) / sr
    body = np.zeros(n, np.float32)
    for mult, amp, rate in ((1.0, 1.00, 5.5), (2.0, 0.16, 9.0),
                            (3.01, 0.05, 14.0)):
        body += (np.sin(2 * np.pi * freq * mult * t)
                 * amp * np.exp(-t * rate)).astype(np.float32)
    rng = np.random.default_rng(seed)
    thud = rng.standard_normal(n).astype(np.float32)
    k = max(4, int(sr / max(60.0, freq * 1.4)))
    thud = np.convolve(thud, np.ones(k, np.float32) / k,
                       mode="same").astype(np.float32)
    thud = np.convolve(thud, np.ones(k, np.float32) / k,
                       mode="same").astype(np.float32)
    body += thud * 0.22 * np.exp(-t * 34.0).astype(np.float32)
    a = max(4, int(sr * ATTACK_MS / 1000.0))
    a = min(a, n // 3)
    env = np.ones(n, np.float32)
    env[:a] = 0.5 - 0.5 * np.cos(
        np.linspace(0.0, np.pi, a, dtype=np.float32))
    env *= np.exp(-t * 7.5).astype(np.float32)
    tail = max(4, int(n * 0.35))
    env[-tail:] *= np.linspace(1.0, 0.0, tail, dtype=np.float32) ** 2
    x = body * env
    x[0] = 0.0
    x[-1] = 0.0
    peak = float(np.abs(x).max())
    if peak > 0:
        x /= peak
    return x
class Audio:
    def __init__(self, sr: int = SR, block: int = BLOCK,
                 volume: int = VOL_DEFAULT,
                 seed: int | None = None) -> None:
        self.sr = sr
        self.block = block
        self.volume = int(np.clip(volume, VOL_MIN, VOL_MAX))
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.stream = None
        self.failed = False
        self.gain = _curve(self.volume)
        self.target = _curve(self.volume)
        self.voices: list[list] = []
        self.clipped = 0
        self.underflow = 0
        self.bank = [_click(sr, CLICK_MS, seed=i, freq=f)
                     for i, f in enumerate(TONES)]
        self.soft = _click(sr, CLICK_MS * 3.4, seed=91, freq=DONE_TONE)
        self.ready = threading.Event()
        self.ready.set()
    @property
    def level_name(self) -> str:
        return "off" if self.volume <= VOL_MIN else str(self.volume)
    @property
    def on(self) -> bool:
        return self.stream is not None and not self.failed
    def wait(self, timeout: float | None = None) -> bool:
        return True
    def start(self) -> bool:
        if self.stream is not None or self.failed:
            return self.on
        try:
            import sounddevice as sd
            self.stream = sd.OutputStream(
                samplerate=self.sr, channels=2, dtype="float32",
                blocksize=self.block, latency="low",
                callback=self._callback)
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
            self.voices = []
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
            self.gain = self.target
        return self.volume
    def _add(self, buf: np.ndarray, gain: float, pan: float) -> None:
        with self.lock:
            if len(self.voices) >= MAX_VOICES:
                self.voices = self.voices[-(MAX_VOICES - 1):]
            self.voices.append([buf, 0, gain, float(np.clip(pan, -1, 1))])
    def place(self, done_frac: float = 0.0, pan: float = 0.0) -> None:
        i = int(self.rng.integers(0, len(self.bank)))
        self._add(self.bank[i], CLICK_GAIN, pan)
    def done(self) -> None:
        self._add(self.soft, DONE_GAIN, 0.0)
    def render(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, 2), np.float32)
        with self.lock:
            g = self.gain
            live = []
            for v in self.voices:
                buf, off, vg, pan = v
                take = min(frames, len(buf) - off)
                if take > 0:
                    seg = buf[off:off + take] * (vg * g)
                    lg = float(np.sqrt(0.5 * (1.0 - pan)))
                    rg = float(np.sqrt(0.5 * (1.0 + pan)))
                    out[:take, 0] += seg * lg * 1.41
                    out[:take, 1] += seg * rg * 1.41
                    v[1] = off + take
                    if v[1] < len(buf):
                        live.append(v)
            self.voices = live
        peak = float(np.abs(out).max())
        if peak > 1.0:
            self.clipped += 1
            out *= 1.0 / peak
        return out
    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self.underflow += 1
        try:
            outdata[:] = self.render(frames)
        except Exception:
            outdata.fill(0.0)
