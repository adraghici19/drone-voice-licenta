"""Analyze the 4 known-angle captures: find the channel->position mapping
(centre channel, handedness, rotation offset) that best explains the 4 angles,
and report whether the array resolves direction."""
import os, glob
import numpy as np
import soundfile as sf
from config import PI_CFG
from doa import SrpPhatDoA, hex_mic_positions, ARRAY_RADIUS

cfg = PI_CFG
capdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doa_captures")

# load captures: true_angle -> (7, N)
caps = {}
for f in sorted(glob.glob(os.path.join(capdir, "angle_*.wav"))):
    ang = float(os.path.basename(f)[6:-4])
    x, sr = sf.read(f, dtype="float32")
    caps[ang] = x.T  # (7, N)
true_angles = sorted(caps)
print("captures:", true_angles)
for a in true_angles:
    rms = np.sqrt(np.mean(caps[a] ** 2, axis=1) + 1e-12)
    print("  {:5.0f} deg  per-ch RMS: {}".format(a, "  ".join("{:.4f}".format(r) for r in rms)))


def hexpos(handed, radius=ARRAY_RADIUS):
    ang = np.deg2rad(np.arange(0, 360, 60) * handed)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


def ang_diff(a, b):
    return min((a - b) % 360, (b - a) % 360)


def circ_mean(angles_deg):
    r = np.deg2rad(angles_deg)
    return np.rad2deg(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360


best = None
for center in range(7):
    periph = [c for c in range(7) if c != center]
    for handed in (+1, -1):
        doa = SrpPhatDoA(cfg.sample_rate, n_fft=1024, mic_xy=hexpos(handed), grid_deg=3.0)
        raw, confs = {}, {}
        ok = True
        for a in true_angles:
            frames = np.stack([caps[a][c] for c in periph])
            az, srp = doa.estimate(frames)
            if az is None:
                ok = False; break
            raw[a] = az
            s0 = srp - srp.min(); confs[a] = float(s0.max() / (s0.mean() + 1e-9))
        if not ok:
            continue
        offsets = [(a - raw[a]) % 360 for a in true_angles]
        rot = circ_mean(offsets)
        errs = [ang_diff((raw[a] + rot) % 360, a) for a in true_angles]
        total = sum(errs)
        if best is None or total < best[0]:
            best = (total, center, periph, handed, rot, raw, errs, confs)

total, center, periph, handed, rot, raw, errs, confs = best
print("\nBest mapping: centre=ch{}  peripheral={}  handed={}  rot={:.0f} deg".format(
    center, periph, handed, rot))
print("mean confidence: {:.1f}".format(np.mean(list(confs.values()))))
print("\n true   estimated   error")
for a in true_angles:
    print("  {:5.0f}   {:6.0f}     {:5.0f} deg".format(a, (raw[a] + rot) % 360, ang_diff((raw[a] + rot) % 360, a)))
print("\nmean angular error: {:.0f} deg".format(np.mean(errs)))
print("VERDICT:", "direction resolved" if np.mean(errs) < 45 else
      "WEAK / not reliably resolved (small array, low level)")
