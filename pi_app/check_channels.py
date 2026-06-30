"""
UMA-8 channel checker.

Captures a few seconds of multichannel audio and reports per-channel energy so
you can see, for THIS specific unit:
  * which channels are live MEMS mics (have signal)
  * which channel is the empty spare PDM port (near-silent)
  * whether any mic is defective (e.g. the thesis noted channel 5 was dead)

Talk / clap continuously while it records.

Usage:
    python check_channels.py                 # auto-find the miniDSP device
    python check_channels.py --device 25     # force a device index
    python check_channels.py --seconds 5
    python check_channels.py --list          # list devices and exit
"""

import argparse
import numpy as np
import sounddevice as sd


def find_device(hint="micarray"):
    cands = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= 7 and (
            "uma" in d["name"].lower() or "micarray" in d["name"].lower()
            or "minidsp" in d["name"].lower()
        ):
            cands.append((i, d))
    if not cands:
        raise RuntimeError("No miniDSP / UMA-8 device found. Use --list to inspect.")
    # prefer a WASAPI endpoint at 48 kHz if present
    for i, d in cands:
        if "wasapi" in sd.query_hostapis(d["hostapi"])["name"].lower():
            return i
    return cands[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    dev = args.device if args.device is not None else find_device()
    info = sd.query_devices(dev)
    nch = info["max_input_channels"]
    host = sd.query_hostapis(info["hostapi"])["name"]
    print("Device [{}] {}  ({} ch, host {})".format(dev, info["name"], nch, host))
    print("Recording {:.0f} s — TALK or CLAP continuously now...".format(args.seconds))

    rec = sd.rec(int(args.seconds * args.samplerate), samplerate=args.samplerate,
                 channels=nch, dtype="float32", device=dev)
    sd.wait()
    print("done.\n")

    rms = np.sqrt(np.mean(rec ** 2, axis=0) + 1e-12)
    peak = np.abs(rec).max(axis=0)
    ref = rms.max()
    print("ch   RMS        peak     rel(dB)   status")
    print("-" * 48)
    for c in range(nch):
        rel_db = 20 * np.log10(rms[c] / (ref + 1e-12) + 1e-12)
        if rms[c] < 1e-4:
            status = "SILENT (spare or dead)"
        elif rel_db < -25:
            status = "very weak (suspect)"
        else:
            status = "live mic"
        print("{:2d}  {:.5f}   {:.4f}   {:6.1f}   {}".format(c, rms[c], peak[c], rel_db, status))

    live = [c for c in range(nch) if rms[c] >= 1e-4 and
            20 * np.log10(rms[c] / (ref + 1e-12) + 1e-12) >= -25]
    print("\nLive channels: {}  (expected 7 MEMS mics)".format(live))
    print("Set pi_app/config.py :: first_mic_channel so logical mics 0..n map onto these.")


if __name__ == "__main__":
    main()
