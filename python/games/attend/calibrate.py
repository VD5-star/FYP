import numpy as np

import psycho

SR = psycho.SR
n = SR * 4
t = np.arange(n) / SR

cases = {}

cases["pure 220Hz (smooth)"] = np.sin(2 * np.pi * 220 * t)

carrier = np.sin(2 * np.pi * 1000 * t)
cases["1kHz, 70Hz AM (very rough)"] = carrier * (1 + 0.9 * np.sin(
    2 * np.pi * 70 * t))
cases["1kHz, 4Hz AM (slow, calm)"] = carrier * (1 + 0.9 * np.sin(
    2 * np.pi * 4 * t))

cases["220+233Hz beating (rough)"] = (np.sin(2 * np.pi * 220 * t)
                                      + np.sin(2 * np.pi * 233 * t))
cases["220+330Hz fifth (smooth)"] = (np.sin(2 * np.pi * 220 * t)
                                     + np.sin(2 * np.pi * 330 * t))

rng = np.random.default_rng(1)
cases["white noise (hissy)"] = rng.standard_normal(n)
pink = np.fft.irfft(np.fft.rfft(rng.standard_normal(n))
                    / np.maximum(np.fft.rfftfreq(n, 1 / SR), 1) ** 0.5, n)
cases["pink noise (calm)"] = pink
brown = np.fft.irfft(np.fft.rfft(rng.standard_normal(n))
                     / np.maximum(np.fft.rfftfreq(n, 1 / SR), 1), n)
cases["brown noise (very calm)"] = brown

cases["buzzer 400Hz saw (harsh)"] = 2 * (t * 400 % 1) - 1

print(f"{'signal':32} {'sharp':>7} {'rough':>8} {'treble%':>8} {'harsh%':>7}")
print("-" * 66)
for name, x in cases.items():
    x = (x / (np.abs(x).max() + 1e-12)).astype(np.float32)
    r = psycho.report(name, x)
    print(f"{name:32} {r['sharp']:7.2f} {r['rough']:8.2f} "
          f"{r['treble']:8.1f} {r['harsh']:7.1f}")

print()
print("a good roughness metric should rank:")
print("  70Hz AM  >>  beating  >>  4Hz AM  ~  pure tone")
