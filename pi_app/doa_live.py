"""
Live direction-of-arrival on the real UMA-8 array.

Reality check: the UMA-8 is a 9 cm array. At speech frequencies its angular
resolution is coarse, and on weak/noisy signals the bearing will be rough
(expect left/right/front/back and maybe +-15-30 deg, not precision). This tool
is built to *try* it on real data and to make the result as good as the array
allows, via an automatic calibration that resolves the unknown channel->mic-
position mapping.

Two unknowns about the hardware are handled by calibration:
  * which of the 7 live channels is the CENTRE mic (the other 6 are the hexagon),
  * the rotation and handedness of the hexagon relative to the device channels.

Workflow (with the array connected):
  1. Put a sound source at a KNOWN bearing (e.g. directly in front = 0 deg),
     playing speech or claps continuously.
  2. Calibrate:   python doa_live.py --calibrate --true-angle 0 --device 25
     This brute-forces the channel mapping that best explains the known bearing
     and saves it to doa_calib.json.
  3. Run live:    python doa_live.py --device 25
     The saved calibration is loaded automatically; a text compass shows the
     estimated bearing and a confidence value.

Without calibration it still runs with a default mapping, but absolute angles
may be rotated/mirrored; relative motion (source moving left<->right) should
still be visible.
"""

import argparse, json, os, itertools
import numpy as np

from config import PI_CFG
from doa import SrpPhatDoA, hex_mic_positions, C_SOUND, ARRAY_RADIUS

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doa_calib.json")


# geometry for a given channel assignment

def positions_for(periph_channels, handed=+1, rot_deg=0.0):
    """6 peripheral mics at 0,60,... (optionally mirrored/rotated). Returns (6,2)."""
    base = np.arange(0, 360, 60) * handed + rot_deg
    ang = np.deg2rad(base)
    return np.stack([ARRAY_RADIUS * np.cos(ang), ARRAY_RADIUS * np.sin(ang)], axis=1)


# capture

def capture(device, seconds, cfg):
    import sounddevice as sd
    if device is None:
        from audio import find_uma8
        device = find_uma8("micArray")
    rec = sd.rec(int(seconds * cfg.sample_rate), samplerate=cfg.sample_rate,
                 channels=cfg.num_device_channels, dtype="float32", device=device)
    sd.wait()
    return rec.T  # (channels, N)


# SRP confidence

def srp_confidence(srp):
    srp = srp - srp.min()
    peak = srp.max()
    return float(peak / (srp.mean() + 1e-9))  # higher = sharper peak


def estimate_with(frames_7, periph_idx, center_idx, handed, rot, cfg):
    """Run SRP-PHAT on the 6 peripheral channels; rot is applied as an output
    offset added to the raw azimuth (matches the multi-angle calibration)."""
    mic_xy = positions_for(periph_idx, handed, 0.0)
    doa = SrpPhatDoA(cfg.sample_rate, n_fft=1024, mic_xy=mic_xy, grid_deg=3.0)
    frames = np.stack([frames_7[c] for c in periph_idx])
    az, srp = doa.estimate(frames)
    if az is None:
        return None, srp
    return (az + rot) % 360.0, srp


# calibration: brute-force the mapping that matches a known angle

def calibrate(frames_7, true_angle, live_channels, cfg):
    best = None
    for center in live_channels:                       # which channel is the centre
        periph = [c for c in live_channels if c != center]
        if len(periph) != 6:
            continue
        for handed in (+1, -1):                        # CW / CCW
            for rot in range(0, 360, 10):              # hexagon rotation
                az, srp = estimate_with(frames_7, periph, center, handed, rot, cfg)
                if az is None:
                    continue
                err = min((az - true_angle) % 360, (true_angle - az) % 360)
                conf = srp_confidence(srp)
                score = err - 0.5 * conf               # prefer low error + sharp peak
                if best is None or score < best[0]:
                    best = (score, center, periph, handed, rot, az, err, conf)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--true-angle", type=float, default=0.0)
    ap.add_argument("--loops", type=int, default=0, help="live: number of updates (0 = until Ctrl+C)")
    args = ap.parse_args()
    cfg = PI_CFG
    live_channels = list(range(7))   # channels 0..6 are the live mics; 7 is the spare

    if args.calibrate:
        print("Calibrating: put the source at {:.0f} deg and keep it sounding...".format(args.true_angle))
        frames = capture(args.device, max(args.seconds, 3.0), cfg)
        best = calibrate(frames, args.true_angle, live_channels, cfg)
        if best is None:
            print("Calibration failed (no signal?)."); return
        _, center, periph, handed, rot, az, err, conf = best
        json.dump({"center": center, "periph": periph, "handed": handed,
                   "rot": rot, "true_angle": args.true_angle},
                  open(CALIB_PATH, "w"), indent=2)
        print("Best mapping: centre=ch{}, peripheral={}, handed={}, rot={} deg".format(
            center, periph, handed, rot))
        print("  -> estimated {:.0f} deg (true {:.0f}, err {:.0f} deg), confidence {:.1f}".format(
            az, args.true_angle, err, conf))
        print("Saved to {}. Now run:  python doa_live.py --device {}".format(CALIB_PATH, args.device or ""))
        return

    # live mode
    if os.path.exists(CALIB_PATH):
        c = json.load(open(CALIB_PATH))
        periph, handed, rot = c["periph"], c["handed"], c["rot"]
        print("Loaded calibration:", c)
    else:
        periph, handed, rot = [0, 1, 2, 3, 4, 5], +1, 0
        print("No calibration found; using default mapping (absolute angle may be off).")

    print("Live DoA — Ctrl+C to stop.\n")
    i = 0
    try:
        while args.loops == 0 or i < args.loops:
            frames = capture(args.device, args.seconds, cfg)
            az, srp = estimate_with(frames, periph, None, handed, rot, cfg)
            conf = srp_confidence(srp)
            if az is None:
                print("  (no audio)"); continue
            # text compass
            dirs = {0: "E", 45: "NE", 90: "N", 135: "NW", 180: "W",
                    225: "SW", 270: "S", 315: "SE"}
            nearest = min(dirs, key=lambda d: min((az - d) % 360, (d - az) % 360))
            bar = int((az / 360) * 32)
            comp = "[" + "-" * bar + "|" + "-" * (32 - bar) + "]"
            flag = "weak" if conf < 3 else ("ok" if conf < 8 else "strong")
            print("  bearing {:5.0f} deg  ~{:2s}  {}  conf {:4.1f} ({})".format(
                az, dirs[nearest], comp, conf, flag))
            i += 1
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
