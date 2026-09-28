import numpy as np

import sounds
from soundcheck import feats

SR = sounds.SR
bank = sounds.render_all()
NAMES = sounds.NAMES

M = {n: feats(bank[n]) for n in NAMES}
mu = np.mean(np.stack([M[n] for n in NAMES]), axis=0)
sd = np.std(np.stack([M[n] for n in NAMES]), axis=0) + 1e-9
Z = {n: (M[n] - mu) / sd for n in NAMES}

print("full-sound distances from bowl:")
for n in NAMES:
    print(f"   {n:8}: {float(np.linalg.norm(Z['bowl'] - Z[n])):.2f}")

print()
print("bowl 2s slices:")
x = bank["bowl"]
seg = SR * 2
for i in range(0, len(x) - seg, seg):
    part = x[i:i + seg]
    if float(np.sqrt((part ** 2).mean())) < 0.02 * 0.115:
        print(f"   slice {i//SR}s: silent")
        continue
    f = (feats(part) - mu) / sd
    ds = {n: float(np.linalg.norm(f - Z[n])) for n in NAMES}
    best = min(ds, key=ds.get)
    print(f"   slice {i//SR:2d}s -> closest {best:8} ({ds[best]:.2f})  "
          f"bowl {ds['bowl']:.2f}")

print()
print("what the slice looks like:")
for i in range(0, len(x) - seg, seg):
    part = x[i:i + seg]
    r = float(np.sqrt((part ** 2).mean()))
    peak = float(np.abs(part).max())
    print(f"   {i//SR:2d}s: rms {r:.4f} peak {peak:.4f}")
