import sys, os
sys.path.insert(0, '..'); sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, soundfile as sf, librosa
import onnxruntime as ort
from config import CFG
from data import extract_features, _drone_noise, _add_fullband

sess = ort.InferenceSession('exported/litevoicenet.onnx', providers=['CPUExecutionProvider'])
h0 = np.zeros((CFG.model.gru_layers, 1, CFG.model.gru_hidden), np.float32)
names = CFG.model.kws_classes
sr = CFG.signal.sample_rate_raw
n = (CFG.synth.seq_frames - 1) * CFG.signal.hop_size + CFG.signal.n_fft


def feat_predict(features):
    p = np.exp(sess.run(None, {'features': features[None].astype(np.float32), 'h_in': h0})[2][0])
    return p


def voice_pred(audio, add_white=False, white_snr=25):
    rng = np.random.default_rng(0)
    a = np.pad(audio, (0, n - len(audio))) if len(audio) < n else audio[:n]
    if add_white:
        w = rng.standard_normal(n).astype('float32')
        s = np.sqrt(np.mean(a**2)+1e-9); wr = np.sqrt(np.mean(w**2)+1e-9)
        a = a + w*(s/(wr*10**(white_snr/20)))
    drone = _drone_noise(n, sr, rng)
    f, _, _ = extract_features(a, drone, 8.0, CFG.array.mic_pairs, CFG.signal.n_fft, CFG.signal.hop_size, sr, rng)
    return feat_predict(f)


# 1) confusion matrix on test set (via training-consistent pipeline)
print("=== CONFUSION MATRIX (test) rows=true, cols=pred ===")
v = pd.read_csv('data/manifests/test.csv')
conf = np.zeros((4, 4), int)
for lab in range(4):
    grp = v[v.kws_label == lab].sample(min(120, (v.kws_label == lab).sum()), random_state=0)
    for _, r in grp.iterrows():
        x, xs = sf.read(r.speech_path, dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        if xs != sr: x = librosa.resample(x, orig_sr=xs, target_sr=sr)
        x = np.pad(x, (0, n-len(x))) if len(x) < n else x[:n]
        is_voice = bool(r['is_voice']) if 'is_voice' in v.columns else True
        rng = np.random.default_rng(int(r.kws_label)*97 + conf.sum())
        if is_voice:
            xi = _add_fullband(x, rng, False)
            drone = _drone_noise(n, sr, rng)
            f, _, _ = extract_features(xi, drone, 8.0, CFG.array.mic_pairs, CFG.signal.n_fft, CFG.signal.hop_size, sr, rng)
        else:
            drone = _drone_noise(n, sr, rng)
            mix = _add_fullband(x + 0.5*drone*(np.sqrt(np.mean(x**2)+1e-9)/(np.sqrt(np.mean(drone**2)+1e-9))), rng, False)
            from data import extract_features_noise
            f, _, _ = extract_features_noise(mix, CFG.array.mic_pairs, CFG.signal.n_fft, CFG.signal.hop_size, sr, rng)
        conf[lab, int(feat_predict(f).argmax())] += 1
print("           " + "  ".join("{:>8s}".format(nm) for nm in names))
for i, nm in enumerate(names):
    print("  {:8s} ".format(nm) + "  ".join("{:8d}".format(conf[i, j]) for j in range(4)))

# 2) white-noise probe
print("\n=== WHITE-NOISE PROBE (clean speech + white noise) ===")
sp = v[v.kws_label == 2].head(4)
for _, r in sp.iterrows():
    x, xs = sf.read(r.speech_path, dtype='float32')
    if x.ndim > 1: x = x.mean(1)
    if xs != sr: x = librosa.resample(x, orig_sr=xs, target_sr=sr)
    line = []
    for snr in [99, 20, 10]:
        p = voice_pred(x, add_white=(snr != 99), white_snr=snr)
        line.append("{}:{}({:.0f})".format('clean' if snr == 99 else snr, names[int(p.argmax())], p.max()*100))
    print("  " + "  ".join(line))

# 3) real capture
cap = os.path.join('..', 'pi_app', 'last_capture.wav')
if os.path.exists(cap):
    x, xs = sf.read(cap, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    if xs != sr: x = librosa.resample(x, orig_sr=xs, target_sr=sr)
    p = voice_pred(x)
    print("\n=== REAL CAPTURE (TED-Ed 'She misses the shot') ===")
    print("  pred:", {names[i]: round(float(p[i]), 2) for i in range(4)})
