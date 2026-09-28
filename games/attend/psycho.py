from __future__ import annotations
import numpy as np
SR = 44100
def _bark(f):
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)
def sharpness(x: np.ndarray, sr: int = SR) -> float:
    n = min(len(x), sr * 4)
    seg = x[:n]
    sp = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / sr)
    z = _bark(f)
    w = np.where(z < 15.8, 1.0, 0.15 * np.exp(0.42 * (z - 15.8)) + 0.85)
    tot = sp.sum()
    if tot <= 0:
        return 0.0
    return float(0.11 * (sp * z * w).sum() / tot)
BANDS = ((50, 150), (150, 300), (300, 600), (600, 1200),
         (1200, 2400), (2400, 4800), (4800, 9600))
MOD_MIN_HZ = 18.0
MOD_PEAK = 70.0
MOD_WID = 0.65
ARTIFACT_HZ = 8.0
def _band_signal(x: np.ndarray, lo: float, hi: float,
                 sr: int) -> np.ndarray:
    n = len(x)
    sp = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    sp = sp * ((f >= lo) & (f < hi))
    return np.fft.irfft(sp, n)
def roughness(x: np.ndarray, sr: int = SR) -> float:
    n = min(len(x), sr * 4)
    seg = x[:n].astype(np.float64)
    if float(np.abs(seg).max()) <= 1e-12:
        return 0.0
    total = 0.0
    power = float((seg ** 2).mean()) + 1e-15
    for lo, hi in BANDS:
        b = _band_signal(seg, lo, hi, sr)
        bp = float((b ** 2).mean())
        if bp / power < 0.01:
            continue
        env = np.abs(_hilbert_env(b))
        dec = max(1, int(sr / 2000))
        e = env[::dec]
        esr = sr / dec
        mean = float(e.mean()) + 1e-15
        e = e - mean
        sp = np.abs(np.fft.rfft(e * np.hanning(len(e)))) / len(e) * 2.0
        f = np.fft.rfftfreq(len(e), 1.0 / esr)
        w = np.exp(-((np.log(np.maximum(f, 1e-6) / MOD_PEAK)) ** 2)
                   / MOD_WID)
        w[f < MOD_MIN_HZ] = 0.0
        w[f > 300.0] = 0.0
        dep_lo = float((sp * (f < MOD_MIN_HZ)).sum())
        dep_band = float((sp * w).sum())
        dep_tot = float(sp.sum()) + 1e-12
        mod_depth = dep_band / mean
        total += dep_band * (bp / power)
    return float(total * 100.0)
def _hilbert_env(x: np.ndarray) -> np.ndarray:
    n = len(x)
    sp = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(sp * h)
def treble_share(x: np.ndarray, sr: int = SR, cut: float = 1500.0) -> float:
    sp = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(sp[f > cut].sum() / (sp.sum() + 1e-9) * 100.0)
def harsh_band(x: np.ndarray, sr: int = SR) -> float:
    sp = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = (f >= 2000.0) & (f <= 5000.0)
    return float(sp[m].sum() / (sp.sum() + 1e-9) * 100.0)
def slope(x: np.ndarray, sr: int = SR) -> float:
    sp = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = (f >= 80.0) & (f <= 10000.0)
    lf = np.log10(f[m])
    lp = 10 * np.log10(sp[m] + 1e-20)
    A = np.vstack([lf, np.ones_like(lf)]).T
    sol, *_ = np.linalg.lstsq(A, lp, rcond=None)
    return float(sol[0])
def crest(x: np.ndarray) -> float:
    r = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    if r <= 0:
        return 0.0
    return float(np.abs(x).max() / r)
def report(name: str, x: np.ndarray, sr: int = SR) -> dict:
    return {
        "name": name,
        "sharp": sharpness(x, sr),
        "rough": roughness(x, sr),
        "treble": treble_share(x, sr),
        "harsh": harsh_band(x, sr),
        "slope": slope(x, sr),
        "crest": crest(x),
        "tonal": tonality(x, sr),
    }
LIMITS = {
    "sharp": 0.80,
    "treble": 25.0,
    "harsh": 6.0,
}
ROUGH_TONAL = 12.0
ROUGH_NOISE = 45.0
TONAL_CREST = 3.6
def tonality(x: np.ndarray, sr: int = SR) -> float:
    sp = np.abs(np.fft.rfft(x)) ** 2
    if sp.sum() <= 0:
        return 0.0
    p = sp / sp.sum()
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum() / np.log(len(p)))
    return 1.0 - ent
def rough_limit(r: dict) -> float:
    return ROUGH_TONAL if r["tonal"] > 0.62 else ROUGH_NOISE
def verdict(r: dict) -> list[str]:
    bad = []
    for k, lim in LIMITS.items():
        if r[k] > lim:
            bad.append(f"{k} {r[k]:.2f} > {lim}")
    lim = rough_limit(r)
    if r["rough"] > lim:
        bad.append(f"rough {r['rough']:.1f} > {lim}")
    return bad
