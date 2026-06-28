import argparse
import time
import sys
import numpy as np

from config import PI_CFG
from features import build_features
from infer import LiteVoiceNetONNX
from detect import KeywordDetector


def _print_event(kw, vad_active, smoothed, classes):
    bar = "  ".join(
        "{}={:.2f}".format(c, p) for c, p in zip(classes, smoothed)
    )
    flag = "SPEECH" if vad_active else "  --  "
    line = "[{}] {}".format(flag, bar)
    if kw:
        line += "   >>> DETECTED: {} <<<".format(kw.upper())
    print(line)


def run_live(cfg, device):
    from audio import MicArrayStream, find_uma8

    if device is None:
        device = find_uma8()
        print("[main] using device index {}".format(device))

    net = LiteVoiceNetONNX(cfg)
    detector = KeywordDetector(cfg)
    hop_period = cfg.hop_samples / cfg.sample_rate

    print("[main] listening... (Ctrl+C to stop)")
    with MicArrayStream(cfg, device=device) as stream:
        while not stream.has_window():
            time.sleep(0.01)
        while True:
            t0 = time.monotonic()
            window = stream.read_window()
            feats = build_features(window, cfg.mic_pairs, cfg.ref_mic,
                                   cfg.n_fft, cfg.hop_size)
            mask, vad, kws = net.run(feats)
            vad_active = float(vad.mean()) > cfg.vad_threshold
            kw = detector.update(kws, now=time.monotonic())
            _print_event(kw, vad_active, detector.smoothed, cfg.kws_classes)

            sleep = hop_period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)


def run_file(cfg, path):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.T
    if sr != cfg.sample_rate:
        import librosa
        audio = np.stack([librosa.resample(c, orig_sr=sr, target_sr=cfg.sample_rate)
                          for c in audio])
    rows = max({cfg.ref_mic} | {m for p in cfg.mic_pairs for m in p}) + 1
    if audio.shape[0] < rows:
        audio = np.broadcast_to(audio[0], (rows, audio.shape[1])).copy()

    net = LiteVoiceNetONNX(cfg)
    detector = KeywordDetector(cfg)

    N = audio.shape[1]
    win = cfg.window_samples
    hop = cfg.hop_samples
    print("[main] offline run over {:.1f} s of audio".format(N / cfg.sample_rate))
    for start in range(0, max(N - win, 1), hop):
        window = audio[:rows, start: start + win]
        if window.shape[1] < win:
            break
        feats = build_features(window, cfg.mic_pairs, cfg.ref_mic,
                               cfg.n_fft, cfg.hop_size)
        mask, vad, kws = net.run(feats)
        vad_active = float(vad.mean()) > cfg.vad_threshold
        t = start / cfg.sample_rate
        kw = detector.update(kws, now=t)
        if kw or vad_active:
            print("t={:6.2f}s ".format(t), end="")
            _print_event(kw, vad_active, detector.smoothed, cfg.kws_classes)
    print("[main] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--file", type=str, default=None)
    args = ap.parse_args()

    cfg = PI_CFG
    if args.fp32:
        cfg.use_int8 = False

    if args.list:
        from audio import list_devices
        list_devices()
        return

    if args.file:
        run_file(cfg, args.file)
        return

    try:
        run_live(cfg, args.device)
    except KeyboardInterrupt:
        print("\n[main] stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
