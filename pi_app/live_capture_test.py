"""
Live array test — run this yourself in a terminal.

Press ENTER, position the phone playing a sound ~25 cm from the array, and the
script records a few seconds and reports, all from the real microphone array:
  * per-channel signal level (which mics picked it up)
  * VAD  -> is a human voice present?
  * KWS  -> which class (help / stop / speech / noise) and its confidence
  * DoA  -> estimated bearing of the sound (rough, small array)
The recording is saved as last_capture.wav for later analysis.

Run:
    .venv\\Scripts\\python.exe pi_app\\live_capture_test.py --device 25 --seconds 4
"""
import argparse, os, time
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import PI_CFG
from features import build_features
from infer import LiteVoiceNetONNX
from doa import SrpPhatDoA, hex_mic_positions


def find_device():
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= 7 and (
            "uma" in d["name"].lower() or "micarray" in d["name"].lower()):
            return i
    raise RuntimeError("UMA-8 not found; pass --device (see check_channels.py --list)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=4.0)
    args = ap.parse_args()
    cfg = PI_CFG
    dev = args.device if args.device is not None else find_device()
    print("Device:", sd.query_devices(dev)["name"])

    try:
        input("\nPut the phone ~25 cm from the array, then press ENTER to record... ")
    except (EOFError, KeyboardInterrupt):
        return
    for k in (3, 2, 1):
        print("  recording in {}...".format(k)); time.sleep(1)
    print("  >>> RECORDING {:.0f}s <<<".format(args.seconds))
    rec = sd.rec(int(args.seconds * cfg.sample_rate), samplerate=cfg.sample_rate,
                 channels=cfg.num_device_channels, dtype="float32", device=dev)
    sd.wait()
    print("  done.\n")
    sf.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_capture.wav"),
             rec[:, :7], cfg.sample_rate)

    # per-channel level
    rms = np.sqrt(np.mean(rec[:, :7] ** 2, axis=0) + 1e-12)
    print("per-channel RMS (ch0-6):", "  ".join("{:.4f}".format(r) for r in rms))

    # ── model: VAD + KWS over sliding windows ────────────────────────────────
    rows = max({cfg.ref_mic} | {m for p in cfg.mic_pairs for m in p}) + 1
    mics = np.stack([rec[:, cfg.first_mic_channel + i] for i in range(rows)])
    net = LiteVoiceNetONNX(cfg)
    win, hop = cfg.window_samples, cfg.hop_samples
    vad_vals, kws_acc, n = [], np.zeros(len(cfg.kws_classes)), 0
    for s in range(0, max(mics.shape[1] - win, 1), hop):
        w = mics[:, s:s + win]
        if w.shape[1] < win:
            break
        feats = build_features(w, cfg.mic_pairs, cfg.ref_mic, cfg.n_fft, cfg.hop_size)
        _, vad, kws = net.run(feats)
        vad_vals.append(float(vad.mean())); kws_acc += kws; n += 1
    kws_mean = kws_acc / max(n, 1)
    vad_mean = float(np.mean(vad_vals)) if vad_vals else 0.0
    top = int(kws_mean.argmax())

    print("\n=== RESULT ===")
    print("VAD (human voice present?): {:.0f}%  -> {}".format(
        vad_mean * 100, "VOICE" if vad_mean > cfg.vad_threshold else "no voice / environment"))
    print("KWS class: {}  ({:.0f}%)".format(cfg.kws_classes[top], kws_mean[top] * 100))
    print("  all:", {cfg.kws_classes[i]: round(float(kws_mean[i]), 2) for i in range(len(cfg.kws_classes))})

    # ── DoA ──────────────────────────────────────────────────────────────────
    doa = SrpPhatDoA(cfg.sample_rate, n_fft=1024, mic_xy=hex_mic_positions(), grid_deg=3.0)
    az, srp = doa.estimate(np.stack([rec[:, i] for i in range(6)]))
    if az is not None:
        srp0 = srp - srp.min()
        conf = float(srp0.max() / (srp0.mean() + 1e-9))
        print("DoA bearing: {:.0f} deg (confidence {:.1f}; rough on a 9 cm array)".format(az, conf))
    print("\nSaved recording -> pi_app/last_capture.wav")


if __name__ == "__main__":
    main()
