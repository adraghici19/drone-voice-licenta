"""
Hardware-free self-test for the Pi inference pipeline.

Validates, on any machine (no UMA-8 needed):
  1. feature tensor shape matches the ONNX `features` input
  2. if a trained model exists, an end-to-end inference runs and returns
     sensible shapes / probability ranges

Run:  python selftest.py
"""

import os
import numpy as np

from config import PI_CFG
from features import build_features


def test_features():
    cfg = PI_CFG
    rows = max({cfg.ref_mic} | {m for p in cfg.mic_pairs for m in p}) + 1
    rng = np.random.default_rng(0)
    window = rng.standard_normal((rows, cfg.window_samples)).astype(np.float32)

    feats = build_features(window, cfg.mic_pairs, cfg.ref_mic, cfg.n_fft, cfg.hop_size)
    assert feats.shape == (cfg.num_features, cfg.seq_frames, cfg.n_freq_bins), \
        "feature shape {} != expected {}".format(
            feats.shape, (cfg.num_features, cfg.seq_frames, cfg.n_freq_bins))
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "features contain NaN/Inf"
    # sin/cos channels must stay in [-1, 1]
    assert feats[1:].min() >= -1.001 and feats[1:].max() <= 1.001
    print("[ok] features: shape {}  range log-mag[{:.1f},{:.1f}]".format(
        feats.shape, feats[0].min(), feats[0].max()))
    return feats


def test_inference(feats):
    here = os.path.dirname(os.path.abspath(__file__))
    fp32 = os.path.normpath(os.path.join(here, PI_CFG.onnx_fp32))
    int8 = os.path.normpath(os.path.join(here, PI_CFG.onnx_int8))
    if not (os.path.exists(fp32) or os.path.exists(int8)):
        print("[skip] no trained model yet (run training/run_training.py first)")
        return
    from infer import LiteVoiceNetONNX
    net = LiteVoiceNetONNX(PI_CFG)
    mask, vad, kws = net.run(feats)
    assert mask.shape == (PI_CFG.seq_frames, PI_CFG.n_freq_bins)
    assert vad.shape == (PI_CFG.seq_frames,)
    assert kws.shape == (len(PI_CFG.kws_classes),)
    assert abs(kws.sum() - 1.0) < 1e-3, "KWS probs must sum to 1 (got {:.3f})".format(kws.sum())
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    print("[ok] inference: mask{} vad{} kws{}  KWS={}".format(
        mask.shape, vad.shape, kws.shape,
        {c: round(float(p), 3) for c, p in zip(PI_CFG.kws_classes, kws)}))


if __name__ == "__main__":
    print("LiteVoiceNet Pi pipeline self-test")
    feats = test_features()
    test_inference(feats)
    print("DONE")
