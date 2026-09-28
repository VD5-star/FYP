from __future__ import annotations
import numpy as np
SR = 44100
LOOP_SECONDS = 48.0
TARGET_LOUD = 0.115
BLOCK_MS = 400.0
def _n(seconds: float = LOOP_SECONDS, sr: int = SR) -> int:
    return int(round(seconds * sr))
def _bin(freq: float, n: int, sr: int = SR) -> float:
    step = sr / n
    return max(1.0, round(freq / step)) * step
def _shaped(n: int, seed: int, shape, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n // 2 + 1
    f = np.fft.rfftfreq(n, 1.0 / sr)
    mag = shape(np.maximum(f, 1e-6)).astype(np.float64)
    mag[0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, half)
    phase[0] = 0.0
    if n % 2 == 0:
        phase[-1] = 0.0
    x = np.fft.irfft(mag * np.exp(1j * phase), n)
    peak = np.abs(x).max()
    return (x / peak).astype(np.float32) if peak > 0 else x.astype(np.float32)
ENV_DECIM = 64
def _smooth_env(n: int, rate: float, depth: float, seed: int,
                sr: int = SR) -> np.ndarray:
    m = max(64, n // ENV_DECIM)
    u = np.linspace(0.0, 1.0, m, endpoint=False, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.zeros(m, np.float64)
    for k in range(2):
        f = _bin(rate * (1.0 + k * 0.6), n, sr)
        cyc = f * n / sr
        out += np.sin(2 * np.pi * cyc * u + rng.uniform(0, 6.283)) / (k + 1)
    peak = np.abs(out).max()
    if peak > 0:
        out /= peak
    env = 1.0 - depth + depth * (0.5 + 0.5 * out)
    xp = np.linspace(0.0, 1.0, m, endpoint=False)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x, xp, env, period=1.0).astype(np.float32)
def _lowpass(x: np.ndarray, cut: float, order: float = 2.0,
             sr: int = SR) -> np.ndarray:
    n = len(x)
    sp = np.fft.rfft(x.astype(np.float64))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    sp *= 1.0 / (1.0 + (np.maximum(f, 1e-6) / cut) ** order)
    return np.fft.irfft(sp, n).astype(np.float32)
def _tone(n: int, freq: float, partials, sr: int = SR,
          seed: int = 0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(seed)
    out = np.zeros(n, np.float64)
    for mult, amp in partials:
        f = _bin(freq * mult, n, sr)
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.283)) * amp
    m = np.abs(out).max()
    return (out / m).astype(np.float32) if m > 0 else out.astype(np.float32)
def _pad(n: int, freq: float, partials, detune: float, rate: float,
         seed: int, sr: int = SR) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.zeros(n, np.float64)
    sways = [_smooth_env(n, rate * (1.0 + k * 0.31), 0.10,
                         seed + k * 17, sr) for k in range(3)]
    for mult, amp in partials:
        base = freq * mult
        for k, d in enumerate((-detune, 0.0, detune)):
            f = _bin(base * (1.0 + d), n, sr)
            cyc = f * n / sr
            out += (np.sin(2 * np.pi * cyc * u + rng.uniform(0, 6.283))
                    * amp * sways[k]) / 3.0
    m = np.abs(out).max()
    return (out / m).astype(np.float32) if m > 0 else out.astype(np.float32)
def _cycles(period: float, seconds: float) -> float:
    return max(1.0, round(seconds / max(0.01, period)))
def _phrase(n: int, period: float, depth: float, floor: float,
            skew: float = 1.0, seconds: float = LOOP_SECONDS,
            sr: int = SR) -> np.ndarray:
    c = _cycles(period, seconds)
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    ph = (t * c) % 1.0
    s = 0.5 - 0.5 * np.cos(2 * np.pi * ph)
    s = s ** skew
    return (floor + (1.0 - floor) * (1.0 - depth + depth * s)).astype(
        np.float32)
def stream(seconds: float = LOOP_SECONDS, sr: int = SR) -> np.ndarray:
    n = _n(seconds, sr)
    x = _pad(n, 587.33, ((1.0, 1.0), (2.0, 0.30), (3.0, 0.13),
                         (4.0, 0.05)), 0.0016, 0.052, 101, sr)
    x = x * _phrase(n, 5.3, 0.55, 0.30, 1.0, seconds, sr)
    return _norm(_lowpass(x, 2200.0, 2.4, sr), sr)
def bowl(seconds: float = LOOP_SECONDS, sr: int = SR) -> np.ndarray:
    n = _n(seconds, sr)
    x = np.zeros(n, np.float32)
    rng = np.random.default_rng(111)
    tones = ((329.63, 0.55), (392.00, 0.42), (440.00, 0.30))
    strikes = max(4, int(round(seconds / 2.4)))
    for k in range(strikes):
        i = int((k + 0.35) * n / strikes) % n
        f0, wobble = tones[k % len(tones)]
        dn = min(n, int(seconds * sr / strikes))
        t = np.arange(dn, dtype=np.float64) / sr
        w = 2 * np.pi * 1.7 * t
        body = np.zeros(dn, np.float64)
        for mult, amp, dec in ((1.0, 1.0, 1.15), (2.40, 0.45, 1.70),
                               (3.85, 0.22, 2.40), (6.20, 0.10, 3.40),
                               (9.10, 0.05, 4.80)):
            f = _bin(f0 * mult, n, sr)
            body += (np.sin(2 * np.pi * f * t + wobble * 0.06 * w)
                     * amp * np.exp(-t * dec))
        a = max(64, int(sr * 0.045))
        env = np.ones(dn, np.float64)
        env[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
        body *= env
        m = np.abs(body).max()
        if m > 0:
            body = body / m
        seg = body.astype(np.float32)
        for start in (i, i - n):
            s, e = start, start + dn
            if e <= 0:
                continue
            s2, e2 = max(0, s), min(n, e)
            if e2 > s2:
                x[s2:e2] += seg[s2 - s:e2 - s] * 0.9
    return _norm(_lowpass(x, 1500.0, 2.8, sr), sr)
def breeze(seconds: float = LOOP_SECONDS, sr: int = SR) -> np.ndarray:
    n = _n(seconds, sr)
    x = _pad(n, 783.99, ((1.0, 1.0), (2.0, 0.03), (3.0, 0.006)),
             0.0026, 0.041, 121, sr)
    x = x * _phrase(n, 8.0, 0.70, 0.22, 1.7, seconds, sr)
    return _norm(_lowpass(x, 1150.0, 3.8, sr), sr)
def strings(seconds: float = LOOP_SECONDS, sr: int = SR) -> np.ndarray:
    n = _n(seconds, sr)
    x = _pad(n, 220.0, ((1.0, 1.0), (2.0, 0.30), (3.0, 0.14),
                        (4.0, 0.06), (6.0, 0.02)), 0.0015, 0.036, 131, sr)
    x = x * _phrase(n, 16.0, 0.42, 0.45, 2.4, seconds, sr)
    return _norm(_lowpass(x, 2400.0, 2.2, sr), sr)
def deep(seconds: float = LOOP_SECONDS, sr: int = SR) -> np.ndarray:
    n = _n(seconds, sr)
    x = _pad(n, 65.41, ((1.0, 1.0), (2.0, 0.26), (3.0, 0.09),
                        (4.0, 0.03)), 0.0011, 0.028, 141, sr)
    x = x * _phrase(n, 10.7, 0.30, 0.60, 1.0, seconds, sr)
    return _norm(_lowpass(x, 520.0, 2.4, sr), sr)
def loudness(x: np.ndarray, sr: int = SR) -> float:
    n = max(1, int(sr * BLOCK_MS / 1000.0))
    hop = max(1, n // 4)
    if len(x) < n:
        return float(np.sqrt((x ** 2).mean()))
    p = np.array([float((x[i:i + n] ** 2).mean())
                  for i in range(0, len(x) - n + 1, hop)])
    if p.size == 0:
        return float(np.sqrt((x ** 2).mean()))
    loud = p[p > p.max() * 0.01]
    if loud.size == 0:
        loud = p
    return float(np.sqrt(loud.mean()))
def _soft_limit(x: np.ndarray, ceiling: float = 0.90) -> np.ndarray:
    knee = ceiling * 0.60
    a = np.abs(x)
    over = a > knee
    if not over.any():
        return x
    room = ceiling - knee
    out = x.copy()
    out[over] = np.sign(x[over]) * (knee + room
                                    * np.tanh((a[over] - knee) / room))
    return out.astype(np.float32)
def _norm(x: np.ndarray, sr: int = SR) -> np.ndarray:
    x = x.astype(np.float32)
    x -= float(x.mean())
    for _ in range(2):
        lo = loudness(x, sr)
        if lo > 0:
            x = x * (TARGET_LOUD / lo)
        x = _soft_limit(x)
    return x.astype(np.float32)
VOICES = {
    "stream": stream,
    "bowl": bowl,
    "breeze": breeze,
    "strings": strings,
    "deep": deep,
}
NAMES = list(VOICES)
LABELS = {
    "stream": "the stream",
    "bowl": "the bowl",
    "breeze": "the breeze",
    "strings": "the strings",
    "deep": "the deep note",
}
PANS = {
    "stream": -0.50,
    "bowl": 0.45,
    "breeze": 0.62,
    "strings": -0.28,
    "deep": 0.00,
}
def render(name: str, seconds: float = LOOP_SECONDS,
           sr: int = SR) -> np.ndarray:
    return VOICES[name](seconds, sr)
def render_all(seconds: float = LOOP_SECONDS,
               sr: int = SR) -> dict[str, np.ndarray]:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(NAMES)) as ex:
        got = list(ex.map(lambda n: render(n, seconds, sr), NAMES))
    return dict(zip(NAMES, got))
