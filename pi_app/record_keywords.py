"""
Interactive keyword recorder for the UMA-8 — run this YOURSELF in a terminal.

It records you saying "help" and "help me" several times through the mic array
and saves clean clips that generate_keywords.py will pick up for (re)training.
This is the step that makes the model recognise YOUR voice for the thesis demo.

Run (from the project root, in your own terminal so you can see the prompts):

    .venv\Scripts\python.exe pi_app\record_keywords.py

Options:
    --device 25       force the UMA-8 device index (see check_channels.py --list)
    --reps 20         how many clips per keyword (default 20)
    --seconds 1.5     length of each recording (default 1.5 s)

For each clip: press ENTER, then say the word once, clearly, ~30 cm from the array.
The recording uses mic channel 0 (logical mic 0) and saves a mono WAV.
"""

import argparse
import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import PI_CFG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {
    "help":    os.path.join(HERE, "..", "training", "data", "tts_keywords", "help"),
    "help me": os.path.join(HERE, "..", "training", "data", "tts_keywords", "help_me"),
}


def find_device():
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= 7 and (
            "uma" in d["name"].lower() or "micarray" in d["name"].lower()):
            for j, dd in enumerate(sd.query_devices()):
                pass
            return i
    raise RuntimeError("UMA-8 not found. Pass --device (see check_channels.py --list).")


def record_one(device, seconds, sr, n_ch, mic_ch):
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=n_ch,
                 dtype="float32", device=device)
    sd.wait()
    return rec[:, mic_ch].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--only", choices=["help", "help_me"], default=None,
                    help="record only one keyword (e.g. --only help_me)")
    args = ap.parse_args()
    cfg = PI_CFG

    device = args.device if args.device is not None else find_device()
    info = sd.query_devices(device)
    print("Recording from [{}] {}".format(device, info["name"]))
    print("Say each word ONCE, clearly, ~30 cm from the array.\n")

    items = OUT.items()
    if args.only:
        want = "help" if args.only == "help" else "help me"
        items = [(w, d) for w, d in OUT.items() if w == want]

    for word, out_dir in items:
        os.makedirs(out_dir, exist_ok=True)
        print("=" * 50)
        print("Keyword: '{}'  —  {} clips".format(word.upper(), args.reps))
        print("=" * 50)
        i = 0
        while i < args.reps:
            try:
                input("  clip {}/{}: press ENTER, then say '{}'... ".format(i + 1, args.reps, word))
            except (EOFError, KeyboardInterrupt):
                print("\nstopped.")
                sys.exit(0)
            sig = record_one(device, args.seconds, cfg.sample_rate,
                             cfg.num_device_channels, cfg.first_mic_channel)
            peak = float(np.abs(sig).max())
            if peak < 0.005:
                print("    too quiet (peak {:.4f}) — speak louder / closer. Retrying.".format(peak))
                continue
            fn = os.path.join(out_dir, "rec_{}_{:02d}.wav".format(word.replace(" ", "_"), i))
            sf.write(fn, sig, cfg.sample_rate)
            print("    saved (peak {:.3f}) -> {}".format(peak, os.path.basename(fn)))
            i += 1

    print("\nAll done. Next:")
    print("  .venv\\Scripts\\python.exe training\\generate_keywords.py")
    print("  .venv\\Scripts\\python.exe training\\run_training.py")


if __name__ == "__main__":
    main()
