"""
Export ONNX (FP32 + INT8) from a saved best-checkpoint, without retraining.

Use this to salvage a model if run_training.py crashes after the checkpoint was
written (run_training.py saves exported/litevoicenet_best.pt on every val
improvement), or to re-export with different settings.

Run:  python export_from_checkpoint.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import pandas as pd
import soundfile as sf
import librosa

from config import CFG
from model import build_model
from data import extract_features, _drone_noise
import export as exp_mod

CKPT = os.path.join(CFG.export_dir, 'litevoicenet_best.pt')


def build_calib_features(n=200):
    """Real feature windows from the val set, class-balanced — for INT8 calibration."""
    v = pd.read_csv(os.path.join(CFG.data_dir, 'manifests', 'val.csv'))
    sr = CFG.signal.sample_rate_raw
    n_samples = (CFG.synth.seq_frames - 1) * CFG.signal.hop_size + CFG.signal.n_fft
    rng = np.random.default_rng(0)
    per = max(1, n // CFG.model.num_kws_classes)
    rows = pd.concat([v[v.kws_label == c].sample(min(per, int((v.kws_label == c).sum())),
                                                  random_state=c)
                      for c in range(CFG.model.num_kws_classes)])
    feats = []
    for _, r in rows.iterrows():
        x, fsr = sf.read(r.speech_path, dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        if fsr != sr: x = librosa.resample(x, orig_sr=fsr, target_sr=sr)
        x = np.pad(x, (0, n_samples - len(x))) if len(x) < n_samples else x[:n_samples]
        noise = _drone_noise(n_samples, sr, rng)
        snr = float(rng.uniform(CFG.synth.snr_min_db, CFG.synth.snr_max_db))
        f, _, _ = extract_features(x, noise, snr, CFG.array.mic_pairs,
                                   CFG.signal.n_fft, CFG.signal.hop_size, sr, rng)
        feats.append(f)
    return np.stack(feats).astype(np.float32)


def main():
    if not os.path.exists(CKPT):
        raise FileNotFoundError("No checkpoint at {}. Run training first.".format(CKPT))
    ck = torch.load(CKPT, map_location='cpu')
    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    epoch = ck.get('epoch', '?') if isinstance(ck, dict) else '?'
    print("Loaded checkpoint (epoch {}) from {}".format(epoch, CKPT))

    model = build_model(CFG)
    model.load_state_dict(state)
    model.eval()

    onnx_path = os.path.join(CFG.export_dir, 'litevoicenet.onnx')
    exp_mod.export_onnx(model, CFG, output_path=onnx_path, seq_frames=CFG.synth.seq_frames)

    print("Building real calibration features for INT8 ...")
    calib = build_calib_features(200)
    print("  calib:", calib.shape, "range [{:.1f}, {:.1f}]".format(calib.min(), calib.max()))
    int8_path = os.path.join(CFG.export_dir, 'litevoicenet_int8.onnx')
    exp_mod.quantize_onnx_int8(onnx_path, output_path=int8_path, cfg=CFG, calib_features=calib)
    print("DONE. Exported FP32 + INT8 from checkpoint.")


if __name__ == "__main__":
    main()
