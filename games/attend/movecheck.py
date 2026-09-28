import numpy as np

import sounds

bank = sounds.render_all()
sr = sounds.SR
win = int(sr * 1.0)

print("how much each sound moves over the loop")
print(f"{'sound':9}{'quiet':>9}{'loud':>9}{'swing':>9}")
fails = []
for n in sounds.NAMES:
    x = bank[n]
    e = np.array([float(np.sqrt((x[i:i + win] ** 2).mean()))
                  for i in range(0, len(x) - win, win // 2)])
    lo, hi = float(np.percentile(e, 10)), float(np.percentile(e, 90))
    swing = hi / max(1e-9, lo)
    print(f"{n:9}{lo:9.4f}{hi:9.4f}{swing:9.2f}x")
    if swing < 1.15:
        fails.append(f"{n} barely moves ({swing:.2f}x)")

mix = sum(bank[n] for n in sounds.NAMES)
me = np.array([float(np.sqrt((mix[i:i + win] ** 2).mean()))
               for i in range(0, len(mix) - win, win // 2)])
mswing = float(np.percentile(me, 90)) / max(1e-9, float(np.percentile(me, 10)))
print(f"\nwhole mix swing  {mswing:.2f}x")

d = {}
for n in sounds.NAMES:
    x = bank[n]
    e = np.array([float(np.sqrt((x[i:i + win] ** 2).mean()))
                  for i in range(0, len(x) - win, win // 2)])
    d[n] = (e - e.mean()) / max(1e-9, e.std())

worst = 0.0
for i, a in enumerate(sounds.NAMES):
    for b in sounds.NAMES[i + 1:]:
        c = abs(float((d[a] * d[b]).mean()))
        worst = max(worst, c)
        if c > 0.75:
            fails.append(f"{a} and {b} move together ({c:.2f})")
print(f"most locked-together pair  {worst:.2f}")

print()
print("sounds move independently" if not fails else "FAILED " + "; ".join(fails))
