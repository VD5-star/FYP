import numpy as np

import psycho
import sounds

SR = psycho.SR
x = sounds.deep().astype(np.float64)[:SR * 6]
power = float((x ** 2).mean())
print("deep, total roughness", round(psycho.roughness(sounds.deep()), 1))
print()
for lo, hi in psycho.BANDS:
    b = psycho._band_signal(x, lo, hi, SR)
    bp = float((b ** 2).mean())
    share = bp / power * 100
    if share < 0.2:
        continue
    env = np.abs(psycho._hilbert_env(b))
    dec = max(1, int(SR / 2000))
    e = env[::dec]
    esr = SR / dec
    mean = float(e.mean()) + 1e-15
    e = e - mean
    sp = np.abs(np.fft.rfft(e * np.hanning(len(e)))) / len(e) * 2.0
    f = np.fft.rfftfreq(len(e), 1.0 / esr)
    w = np.exp(-((np.log(np.maximum(f, 1e-6) / 70.0)) ** 2) / 0.65)
    w[f < 15.0] = 0.0
    w[f > 300.0] = 0.0
    depth = float((sp * w).sum()) / mean
    print(f"  {lo:5}-{hi:5} Hz: {share:5.1f}%  depth {depth:6.2f}")
    peaks = np.argsort(sp[f > 15])[-3:] + 0
    sel = np.nonzero(f > 15.0)[0][peaks]
    for i in sel[:3]:
        print(f"        peak at {f[i]:6.1f} Hz")
