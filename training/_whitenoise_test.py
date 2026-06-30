"""Hypothesis: real array audio has full-band mic self-noise that training
(low-pass drone noise only) never had -> OOD -> help bias.
Test: add white (full-band) noise to clean speech and watch the prediction."""
import sys, os
sys.path.insert(0, '..'); sys.path.insert(0, '.')
import numpy as np, pandas as pd, soundfile as sf, librosa
import onnxruntime as ort
from config import CFG
from data import extract_features, _drone_noise

sess = ort.InferenceSession('exported/litevoicenet.onnx', providers=['CPUExecutionProvider'])
h0 = np.zeros((CFG.model.gru_layers, 1, CFG.model.gru_hidden), np.float32)
names = CFG.model.kws_classes
sr = CFG.signal.sample_rate_raw
n = (CFG.synth.seq_frames - 1) * CFG.signal.hop_size + CFG.signal.n_fft


def predict(audio):
    rng = np.random.default_rng(0)
    a = np.pad(audio, (0, n - len(audio))) if len(audio) < n else audio[:n]
    drone = _drone_noise(n, sr, rng)
    f, _, _ = extract_features(a, drone, 8.0, CFG.array.mic_pairs,
                               CFG.signal.n_fft, CFG.signal.hop_size, sr, rng)
    p = np.exp(sess.run(None, {'features': f[None].astype(np.float32), 'h_in': h0})[2][0])
    return p


v = pd.read_csv('data/manifests/test.csv')
for lab, tag in [(2, 'speech'), (1, 'stop')]:
    print("\n=== clean {} + increasing WHITE noise ===".format(tag))
    for _, r in v[v.kws_label == lab].head(4).iterrows():
        x, xs = sf.read(r.speech_path, dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        if xs != sr: x = librosa.resample(x, orig_sr=xs, target_sr=sr)
        x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
        line = []
        rng = np.random.default_rng(1)
        for snr in [99, 20, 10, 0, -6]:
            if snr == 99:
                xn = x
            else:
                w = rng.standard_normal(len(x)).astype('float32')
                s_rms = np.sqrt(np.mean(x**2)+1e-9); w_rms = np.sqrt(np.mean(w**2)+1e-9)
                xn = x + w * (s_rms/(w_rms*10**(snr/20)))
            p = predict(xn)
            line.append("{}:{}({:.0f})".format(snr if snr != 99 else 'clean', names[int(p.argmax())], p.max()*100))
        print("  " + "  ".join(line))
