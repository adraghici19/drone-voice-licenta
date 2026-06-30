"""
Controlled live test: record N seconds from the UMA-8, slide the model over the
capture, and report the strongest keyword per window + the overall best.

Say one keyword ("help", "help me", or "stop") clearly while it records.

Usage:
    python live_test.py --device 25 --seconds 3
"""

import argparse
import numpy as np
import sounddevice as sd

from config import PI_CFG
from features import build_features
from infer import LiteVoiceNetONNX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()
    cfg = PI_CFG

    if args.device is None:
        from audio import find_uma8
        args.device = find_uma8("micArray")
    print("Recording {:.0f} s from device {} — SAY A KEYWORD now...".format(args.seconds, args.device))

    rec = sd.rec(int(args.seconds * cfg.sample_rate), samplerate=cfg.sample_rate,
                 channels=cfg.num_device_channels, dtype="float32", device=args.device)
    sd.wait()
    print("done.\n")

    # select logical mics 0..max from device channels
    rows = max({cfg.ref_mic} | {m for p in cfg.mic_pairs for m in p}) + 1
    mics = np.stack([rec[:, cfg.first_mic_channel + i] for i in range(rows)])  # (rows, N)

    net = LiteVoiceNetONNX(cfg)
    win = cfg.window_samples
    hop = cfg.hop_samples
    names = cfg.kws_classes
    best = np.zeros(len(names))
    print("window    " + "  ".join("{:>9s}".format(n) for n in names) + "   pred")
    for start in range(0, mics.shape[1] - win + 1, hop):
        w = mics[:, start:start + win]
        feats = build_features(w, cfg.mic_pairs, cfg.ref_mic, cfg.n_fft, cfg.hop_size)
        _, vad, kws = net.run(feats)
        best = np.maximum(best, kws)
        t = start / cfg.sample_rate
        print("t={:4.2f}s   ".format(t) + "  ".join("{:9.2f}".format(p) for p in kws)
              + "   " + names[int(kws.argmax())])

    print("\nBest keyword probability across the capture:")
    for n, p in sorted(zip(names, best), key=lambda x: -x[1]):
        print("  {:10s} {:.2f}".format(n, p))


if __name__ == "__main__":
    main()
