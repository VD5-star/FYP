import psycho
import sounds

bank = sounds.render_all()

print(f"{'sound':8} {'sharp':>6} {'rough':>7} {'limit':>6} {'tonal':>6} "
      f"{'treble%':>8} {'harsh%':>7} {'slope':>7}   problems")
print("-" * 86)
for n in sounds.NAMES:
    r = psycho.report(n, bank[n])
    bad = psycho.verdict(r)
    print(f"{n:8} {r['sharp']:6.2f} {r['rough']:7.1f} "
          f"{psycho.rough_limit(r):6.0f} {r['tonal']:6.2f} "
          f"{r['treble']:8.1f} {r['harsh']:7.1f} {r['slope']:7.1f}   "
          f"{', '.join(bad) if bad else 'ok'}")

print()
print("limits:", psycho.LIMITS)
print()
print("slope: natural calm sound falls about -6 to -12 dB per decade")
print("sharp: acum, above ~1.6 starts to feel piercing")
print("rough: amplitude flutter 20-300Hz, the 'buzzy' feeling")
