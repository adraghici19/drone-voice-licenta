import argparse
import time
import sys
import numpy as np

from config import PI_CFG
from features import build_features
from infer import LiteVoiceNetONNX
from detect import KeywordDetector
from doa import SrpPhatDoA, hex_mic_positions

_ACTIVE_MICS = [0, 1, 2, 3, 4, 6]


def run(cfg, device):
    from audio import MicArrayStream, find_uma8

    if device is None:
        try:
            device = find_uma8()
        except RuntimeError:
            print("[combined] UMA-8 not found, use --device N")
            sys.exit(1)

    net = LiteVoiceNetONNX(cfg)
    detector = KeywordDetector(cfg)
    doa = SrpPhatDoA(cfg.sample_rate, n_fft=512,
                     mic_xy=hex_mic_positions(), grid_deg=5.0,
                     fmin=300.0, fmax=3400.0)

    hop_period = cfg.hop_samples / cfg.sample_rate
    print("[combined] listening... (Ctrl+C to stop)")
    print("{:<10} {:<40} {:<12} {}".format("VAD", "KWS scores", "direction", "event"))
    print("-" * 75)

    with MicArrayStream(cfg, device=device) as stream:
        while not stream.has_window():
            time.sleep(0.01)
        while True:
            t0 = time.monotonic()

            raw = stream._latest(cfg.window_samples)

            needed = sorted({cfg.ref_mic} | {m for p in cfg.mic_pairs for m in p})
            rows = max(needed) + 1
            window = np.zeros((rows, cfg.window_samples), dtype=np.float32)
            for idx in needed:
                window[idx] = raw[cfg.first_mic_channel + idx]

            feats = build_features(window, cfg.mic_pairs, cfg.ref_mic,
                                   cfg.n_fft, cfg.hop_size)
            mask, vad, kws = net.run(feats)
            vad_active = float(vad.mean()) > cfg.vad_threshold
            kw = detector.update(kws, now=time.monotonic())

            az = None
            if vad_active or kw:
                mic6 = raw[_ACTIVE_MICS]
                az, _ = doa.estimate(mic6)

            flag = "SPEECH" if vad_active else "  --  "
            scores = "  ".join("{:s}={:.2f}".format(c, p)
                               for c, p in zip(cfg.kws_classes, detector.smoothed))
            doa_str = "{:5.1f} deg".format(az) if az is not None else "    --   "
            event = ">>> {:s} <<<".format(kw.upper()) if kw else ""
            print("[{:s}] {:s}  {:s}  {:s}".format(flag, scores, doa_str, event))

            sleep = hop_period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.50)
    args = ap.parse_args()

    cfg = PI_CFG
    cfg.kws_threshold = args.threshold
    if args.fp32:
        cfg.use_int8 = False

    try:
        run(cfg, args.device)
    except KeyboardInterrupt:
        print("\n[combined] stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
