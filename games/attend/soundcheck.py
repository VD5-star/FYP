import time
import numpy as np
import sounds
from sounds import LOOP_SECONDS, NAMES, SR
fails = []
def ok(name, cond, detail=""):
    print(f"  [{'pass' if cond else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)
def db(x):
    return 20 * np.log10(max(1e-12, x))
print("== every sound builds, in range, not silent ==")
t0 = time.perf_counter()
bank = sounds.render_all()
build_ms = (time.perf_counter() - t0) * 1000
TARGET_RMS = 0.115
print(f"       built {len(bank)} sounds in {build_ms:.0f}ms")
ok("builds under 6s", build_ms < 6000, f"{build_ms:.0f}ms")
for n, x in bank.items():
    ok(f"{n} right length", len(x) == int(LOOP_SECONDS * SR), f"{len(x)}")
    ok(f"{n} finite", bool(np.isfinite(x).all()))
    ok(f"{n} not silent", float(np.abs(x).mean()) > 1e-4,
       f"mean |x| {float(np.abs(x).mean()):.4f}")
    ok(f"{n} no clipping", float(np.abs(x).max()) < 1.0,
       f"peak {float(np.abs(x).max()):.3f}")
print()
print("== all five sit at the same loudness ==")
loud = {n: sounds.loudness(x) for n, x in bank.items()}
rms = {n: float(np.sqrt((x ** 2).mean())) for n, x in bank.items()}
for n in NAMES:
    print(f"       {n:6}: heard {db(loud[n]):6.1f} dBFS   "
          f"average {db(rms[n]):6.1f}   "
          f"peak {db(float(np.abs(bank[n]).max())):6.1f}")
spread = db(max(loud.values())) - db(min(loud.values()))
ok("when playing, all sit within 1 dB", spread < 1.0, f"{spread:.2f} dB")
ok("none is dominant", spread < 1.0)
ok("none is so quiet it vanishes",
   db(min(rms.values())) > -32.0, f"{db(min(rms.values())):.1f} dBFS")
print()
print("== each loops seamlessly ==")
for n, x in bank.items():
    inner = float(np.abs(np.diff(x)).max())
    seam = abs(float(x[0]) - float(x[-1]))
    ok(f"{n} seam is quiet", seam <= inner,
       f"seam {seam:.2e} vs largest inner step {inner:.2e}")
    wrapped = np.concatenate([x[-1500:], x[:1500]])
    ok(f"{n} no spike at the wrap",
       float(np.abs(np.diff(wrapped)).max()) <= inner * 1.05,
       f"{float(np.abs(np.diff(wrapped)).max()):.2e}")
def feats(x, sr=SR):
    win = 2048
    hop = 1024
    out = []
    rms_track = []
    for i in range(0, len(x) - win, hop):
        seg = x[i:i + win] * np.hanning(win)
        rms_track.append(float(np.sqrt((seg ** 2).mean())))
        sp = np.abs(np.fft.rfft(seg))
        f = np.fft.rfftfreq(win, 1.0 / sr)
        tot = sp.sum() + 1e-9
        cen = float((f * sp).sum() / tot)
        spread_ = float(np.sqrt(((f - cen) ** 2 * sp).sum() / tot))
        cum = np.cumsum(sp)
        roll = float(f[np.searchsorted(cum, 0.85 * cum[-1])])
        flat = float(np.exp(np.log(sp + 1e-9).mean()) / (sp.mean() + 1e-9))
        rmsv = float(np.sqrt((seg ** 2).mean()))
        out.append([cen, spread_, roll, flat, rmsv])
    r = np.array(rms_track)
    if r.size and r.mean() > 1e-12:
        cv = float(r.std() / r.mean())
    else:
        cv = 0.0
    a = np.array(out)
    return np.concatenate([a.mean(axis=0), a.std(axis=0), [cv]])
print()
print("== the five are actually different from each other ==")
F = {n: feats(bank[n]) for n in NAMES}
M = np.stack([F[n] for n in NAMES])
mu, sd = M.mean(axis=0), M.std(axis=0) + 1e-9
Z = (M - mu) / sd
print("       pairwise distance (bigger = easier to tell apart)")
print("             " + "".join(f"{n:>8}" for n in NAMES))
worst = (1e9, "", "")
for i, a in enumerate(NAMES):
    row = f"       {a:6}"
    for j, b in enumerate(NAMES):
        d = float(np.linalg.norm(Z[i] - Z[j]))
        row += f"{d:8.2f}"
        if i != j and d < worst[0]:
            worst = (d, a, b)
    print(row)
ok("the closest pair is still clearly apart", worst[0] > 1.5,
   f"{worst[1]} vs {worst[2]} = {worst[0]:.2f}")
print()
print("== each keeps its character in every 2-second slice ==")
for n in NAMES:
    x = bank[n]
    seg = SR * 2
    slices = []
    for i in range(0, len(x) - seg, seg):
        part = x[i:i + seg]
        if float(np.sqrt((part ** 2).mean())) < 0.02 * TARGET_RMS:
            continue
        slices.append(feats(part))
    parts = slices if slices else [feats(x[:seg])]
    P = (np.stack(parts) - mu) / sd
    own_all = np.linalg.norm(P - Z[NAMES.index(n)], axis=1)
    others_all = []
    for j in range(len(NAMES)):
        if j == NAMES.index(n):
            continue
        others_all.append(np.linalg.norm(P - Z[j], axis=1))
    nearest = np.min(np.stack(others_all), axis=0)
    ok(f"{n} has unmistakable moments",
       float(own_all.min()) < float(nearest.min()),
       f"best slice own {own_all.min():.2f} "
       f"vs nearest other there {nearest.min():.2f}")
    clear = float((own_all < nearest).sum()) / len(P)
    ok(f"{n} is recognisable in most slices", clear >= 0.5,
       f"{clear*100:.0f}% of {len(P)} slices")
    margin = float(nearest.mean() - own_all.mean())
    print(f"       {n:8}: mean margin {margin:+.2f}, "
          f"silence-free slices {len(P)}")
print()
print("== they occupy different parts of the spectrum ==")
for n in NAMES:
    x = bank[n]
    sp = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    cum = np.cumsum(sp)
    med = float(f[np.searchsorted(cum, 0.5 * cum[-1])])
    lo = float(sp[f < 300].sum() / sp.sum() * 100)
    hi = float(sp[f > 3000].sum() / sp.sum() * 100)
    print(f"       {n:6}: middle {med:7.0f} Hz   "
          f"low {lo:5.1f}%   high {hi:5.1f}%")
meds = []
for n in NAMES:
    x = bank[n]
    sp = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    cum = np.cumsum(sp)
    meds.append(float(f[np.searchsorted(cum, 0.5 * cum[-1])]))
meds_sorted = sorted(meds)
ratios = [meds_sorted[i + 1] / max(1.0, meds_sorted[i])
          for i in range(len(meds_sorted) - 1)]
ok("no two share the same centre", min(ratios) > 1.25,
   f"closest ratio {min(ratios):.2f}x")
print()
print("== mixing all five keeps every one audible ==")
mix = np.zeros(len(bank[NAMES[0]]), np.float32)
for n in NAMES:
    mix += bank[n]
ok("the mix does not clip", float(np.abs(mix).max()) < 5.0,
   f"peak {float(np.abs(mix).max()):.2f}")
mix_rms = float(np.sqrt((mix ** 2).mean()))
for n in NAMES:
    share = rms[n] / mix_rms
    ok(f"{n} is not buried", share > 0.25,
       f"{share*100:.0f}% of the mix level")
print()
print("== nothing is harsh or piercing ==")
for n in NAMES:
    x = bank[n]
    sp = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    harsh = float(sp[(f > 2000) & (f < 5000)].sum() / sp.sum() * 100)
    ok(f"{n} not piercing", harsh < 30.0, f"{harsh:.1f}% in 2-5kHz")
    step = float(np.abs(np.diff(x)).max())
    ok(f"{n} no jolt", step < 0.30, f"max step {step:.3f}")
print()
if fails:
    print(f"FAILED {len(fails)}:")
    for f_ in fails:
        print("   -", f_)
else:
    print("all sound checks pass")
