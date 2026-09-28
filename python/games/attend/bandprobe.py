import numpy as np

import psycho
import sounds

SR = psycho.SR
bank = sounds.render_all()

print("where each sound's energy sits, and roughness per band")
print()
for n in ("stream", "breeze", "bowl", "strings"):
    x = bank[n].astype(np.float64)[:SR * 4]
    power = float((x ** 2).mean())
    print(f"{n}:")
    for lo, hi in psycho.BANDS:
        b = psycho._band_signal(x, lo, hi, SR)
        bp = float((b ** 2).mean())
        share = bp / power * 100
        if share < 1.0:
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
        print(f"   {lo:5}-{hi:5} Hz: {share:5.1f}% of energy, "
              f"modulation depth {depth:6.2f}")
    print()

rng = np.random.default_rng(3)
n4 = SR * 4
raw = rng.standard_normal(n4)
sp = np.fft.rfft(raw)
f = np.fft.rfftfreq(n4, 1 / SR)
sp *= 1.0 / (1.0 + (np.maximum(f, 1e-6) / 200.0) ** 2.4)
ref = np.fft.irfft(sp, n4)
ref = ref / np.abs(ref).max()
print("reference: plain lowpassed noise (no shaping at all)")
print(f"   roughness {psycho.roughness(ref.astype(np.float32)):.1f}")
print()
print("this is the floor for any noise-based sound at this bandwidth")
