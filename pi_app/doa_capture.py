"""
Capture the array at a KNOWN angle for DoA analysis. Run it once per angle.

Keep the ARRAY fixed on the table. Move the PHONE around it at a fixed distance
(~30-40 cm), playing a clear CONTINUOUS sound (speech, music, or a tone).
Pick one edge of the array as 'front' = 0 deg and keep it consistent.

    .venv\\Scripts\\python.exe pi_app\\doa_capture.py --angle 0   --device 25
    .venv\\Scripts\\python.exe pi_app\\doa_capture.py --angle 90  --device 25
    .venv\\Scripts\\python.exe pi_app\\doa_capture.py --angle 180 --device 25
    .venv\\Scripts\\python.exe pi_app\\doa_capture.py --angle 270 --device 25

Each capture is saved to  pi_app/doa_captures/angle_<A>.wav  (7 channels).
Tell me when all four are done and I'll analyze them.
"""
import argparse, os, time
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import PI_CFG
from doa import SrpPhatDoA, hex_mic_positions


def find_device():
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= 7 and (
            "uma" in d["name"].lower() or "micarray" in d["name"].lower()):
            return i
    raise RuntimeError("UMA-8 not found; pass --device")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=float, required=True, help="known phone angle in degrees")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=4.0)
    args = ap.parse_args()
    cfg = PI_CFG
    dev = args.device if args.device is not None else find_device()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doa_captures")
    os.makedirs(out, exist_ok=True)

    print("Device:", sd.query_devices(dev)["name"])
    print("Phone at {:.0f} deg, ~30-40 cm, playing a CONTINUOUS sound.".format(args.angle))
    try:
        input("Press ENTER to record... ")
    except (EOFError, KeyboardInterrupt):
        return
    for k in (3, 2, 1):
        print("  {}...".format(k)); time.sleep(1)
    print("  >>> recording {:.0f}s <<<".format(args.seconds))
    rec = sd.rec(int(args.seconds * cfg.sample_rate), samplerate=cfg.sample_rate,
                 channels=cfg.num_device_channels, dtype="float32", device=dev)
    sd.wait()
    path = os.path.join(out, "angle_{:.0f}.wav".format(args.angle))
    sf.write(path, rec[:, :7], cfg.sample_rate)

    rms = np.sqrt(np.mean(rec[:, :7] ** 2, axis=0) + 1e-12)
    print("\n  per-channel RMS (ch0-6):", "  ".join("{:.4f}".format(r) for r in rms))
    # quick uncalibrated SRP-PHAT (default channel order)
    doa = SrpPhatDoA(cfg.sample_rate, n_fft=1024, mic_xy=hex_mic_positions(), grid_deg=3.0)
    az, srp = doa.estimate(np.stack([rec[:, i] for i in range(6)]))
    if az is not None:
        srp0 = srp - srp.min(); conf = float(srp0.max() / (srp0.mean() + 1e-9))
        print("  uncalibrated SRP-PHAT estimate: {:.0f} deg (confidence {:.1f})".format(az, conf))
    print("  saved -> {}".format(path))


if __name__ == "__main__":
    main()
