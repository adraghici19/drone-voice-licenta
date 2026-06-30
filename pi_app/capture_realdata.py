"""
Capture REAL array audio to close the sim-to-real gap, then auto-segment into
~1 s clips for training. Run this YOURSELF in a terminal.

Usage:
    # ~3 min of speech (play TED-Ed / talk near the array, varied):
    .venv\\Scripts\\python.exe pi_app\\capture_realdata.py --label speech --minutes 3 --device 25

    # ~2 min of room/environment (quiet room, or play nature/music; NO speech):
    .venv\\Scripts\\python.exe pi_app\\capture_realdata.py --label noise --minutes 2 --device 25

Clips are saved to  training/data/real_array/<label>/  and picked up by the
manifest builder for the next training run.
"""
import argparse, os, time
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import PI_CFG

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "training", "data", "real_array")


def find_device():
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= 7 and (
            "uma" in d["name"].lower() or "micarray" in d["name"].lower()):
            return i
    raise RuntimeError("UMA-8 not found; pass --device")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["speech", "noise"])
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--clip-s", type=float, default=1.0)
    args = ap.parse_args()
    cfg = PI_CFG
    dev = args.device if args.device is not None else find_device()
    out = os.path.join(OUT_ROOT, args.label)
    os.makedirs(out, exist_ok=True)
    sr = cfg.sample_rate

    print("Device:", sd.query_devices(dev)["name"])
    print("Will record {:.1f} min of '{}'.".format(args.minutes, args.label))
    if args.label == "speech":
        print("  -> play TED-Ed / talk near the array; vary it, keep voice present.")
    else:
        print("  -> room tone / environment / nature / music; NO human speech.")
    try:
        input("Press ENTER to start recording... ")
    except (EOFError, KeyboardInterrupt):
        return
    for k in (3, 2, 1):
        print("  starting in {}...".format(k)); time.sleep(1)

    total = int(args.minutes * 60 * sr)
    print("  >>> RECORDING {:.1f} min — go! <<<".format(args.minutes))
    rec = sd.rec(total, samplerate=sr, channels=cfg.num_device_channels,
                 dtype="float32", device=dev)
    sd.wait()
    print("  done. segmenting...")

    mono = rec[:, cfg.first_mic_channel]            # logical mic 0
    clip_n = int(args.clip_s * sr)
    existing = len([f for f in os.listdir(out) if f.endswith(".wav")])
    saved = 0
    for i, s in enumerate(range(0, len(mono) - clip_n + 1, clip_n)):
        seg = mono[s:s + clip_n]
        peak = float(np.abs(seg).max())
        # skip near-silent slices for speech (keep all for noise/room tone)
        if args.label == "speech" and peak < 0.003:
            continue
        sf.write(os.path.join(out, "real_{}_{:04d}.wav".format(args.label, existing + saved)),
                 seg, sr)
        saved += 1
    print("Saved {} '{}' clips to {} (total now ~{}).".format(
        saved, args.label, out, existing + saved))
    print("Next: rebuild manifests + retrain (I'll do that).")


if __name__ == "__main__":
    main()
