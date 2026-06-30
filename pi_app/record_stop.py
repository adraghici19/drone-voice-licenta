"""
Records you saying "stop" 30 times through the UMA-8 mic array.
Saves clips to training/data/stop_real/ for fine-tuning.

Run from project root:
    .venv\Scripts\python.exe pi_app\record_stop.py --device 25
"""

import argparse
import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import PI_CFG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "training", "data", "stop_real")


def record_one(device, seconds, sr, n_ch, mic_ch):
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=n_ch,
                 dtype="float32", device=device)
    sd.wait()
    return rec[:, mic_ch].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=1.5)
    args = ap.parse_args()
    cfg = PI_CFG

    os.makedirs(OUT_DIR, exist_ok=True)

    device = args.device
    if device is None:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] >= 7 and (
                    "uma" in d["name"].lower() or "micarray" in d["name"].lower()):
                device = i
                break
        if device is None:
            print("UMA-8 not found. Pass --device N")
            sys.exit(1)

    info = sd.query_devices(device)
    print("Recording from [{}] {}".format(device, info["name"]))
    print("Say 'STOP' once clearly after each ENTER (~30 cm from array).\n")
    print("=" * 50)
    print("Keyword: 'STOP'  —  {} clips".format(args.reps))
    print("=" * 50)

    i = 0
    while i < args.reps:
        try:
            input("  clip {}/{}: press ENTER, then say 'stop'... ".format(i + 1, args.reps))
        except (EOFError, KeyboardInterrupt):
            print("\nStopped early ({} clips saved).".format(i))
            sys.exit(0)

        sig = record_one(device, args.seconds, cfg.sample_rate,
                         cfg.num_device_channels, cfg.first_mic_channel)
        peak = float(np.abs(sig).max())
        rms  = float(np.sqrt(np.mean(sig ** 2)))

        if peak < 0.002 or rms < 0.0003:
            print("    too quiet (peak={:.4f} rms={:.4f}) — speak louder. Retrying.".format(peak, rms))
            continue

        fn = os.path.join(OUT_DIR, "stop_real_{:03d}.wav".format(i))
        sf.write(fn, sig, cfg.sample_rate)
        print("    saved (peak={:.3f} rms={:.4f}) -> {}".format(peak, rms, os.path.basename(fn)))
        i += 1

    print("\nDone! {} clips saved to {}".format(i, OUT_DIR))
    print("Next: .venv\\Scripts\\python.exe training\\add_stop_to_manifest.py")


if __name__ == "__main__":
    main()
