"""
Verifies stop_real clips against the ONNX model using the same
feature pipeline as training (48 kHz, drone-noise mix, log-power STFT).
"""
import os, sys
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import CFG
from data import _drone_noise, _stft

STOP_DIR  = os.path.join("data", "stop_real")
ONNX_PATH = os.path.join("exported", "litevoicenet.onnx")

SR        = CFG.signal.sample_rate_raw   # 48000
N_FFT     = CFG.signal.n_fft            # 512
HOP       = CFG.signal.hop_size         # 256
SEQ       = CFG.synth.seq_frames        # 100
N_FREQ    = CFG.signal.n_freq_bins      # 257
N_SAMPLES = (SEQ - 1) * HOP + N_FFT    # 25856  (~0.54s at 48kHz)
CLASSES   = CFG.model.kws_classes
MIC_PAIRS = CFG.array.mic_pairs
ARRAY_R   = 0.045
C         = 343.0

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 2
sess = ort.InferenceSession(ONNX_PATH, sess_options=sess_opts,
                             providers=["CPUExecutionProvider"])

freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
max_tdoa = (2 * ARRAY_R) / C


def build_features(sig: np.ndarray, snr_db: float = 6.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = _drone_noise(N_SAMPLES, SR, rng)
    s_rms = np.sqrt(np.mean(sig ** 2) + 1e-9)
    n_rms = np.sqrt(np.mean(noise ** 2) + 1e-9)
    noise_s = noise * (s_rms / (n_rms * 10 ** (snr_db / 20)))
    mixture = sig + noise_s
    peak = np.abs(mixture).max()
    mixture = mixture * (0.9 / (peak + 1e-9))

    M = _stft(mixture, N_FFT, HOP)           # (257, T)
    T = M.shape[1]
    log_mag = np.log(np.abs(M) ** 2 + 1e-8)  # (257, T)

    channels = [log_mag]
    for _ in MIC_PAIRS:
        tdoa = rng.uniform(-max_tdoa, max_tdoa)
        pd = 2.0 * np.pi * freqs * tdoa
        ipd = pd[:, np.newaxis] * np.ones(T)
        channels.append(np.sin(ipd).astype(np.float32))
        channels.append(np.cos(ipd).astype(np.float32))

    feat = np.stack(channels, axis=0).transpose(0, 2, 1)  # (5, T, 257)
    # pad/crop to SEQ frames
    if feat.shape[1] < SEQ:
        feat = np.pad(feat, ((0,0),(0, SEQ - feat.shape[1]),(0,0)))
    else:
        feat = feat[:, :SEQ, :]
    return feat.astype(np.float32)[np.newaxis]  # (1, 5, 100, 257)


def best_window(sig: np.ndarray) -> np.ndarray:
    """Return the N_SAMPLES window with maximum RMS energy."""
    if len(sig) <= N_SAMPLES:
        return np.pad(sig, (0, N_SAMPLES - len(sig)))
    step = HOP
    best_e, best_s = -1.0, 0
    for s in range(0, len(sig) - N_SAMPLES + 1, step):
        e = float(np.mean(sig[s:s + N_SAMPLES] ** 2))
        if e > best_e:
            best_e, best_s = e, s
    return sig[best_s:best_s + N_SAMPLES].copy()


wavs = sorted(f for f in os.listdir(STOP_DIR) if f.endswith(".wav"))
print("Verifying {} stop_real clips\n".format(len(wavs)))
print("{:<32} {:>6} {:>6} {:>8} {:>6}  top".format("file","help","stop","speech","noise"))
print("-" * 70)

results = []
for fn in wavs:
    sig, sr_orig = sf.read(os.path.join(STOP_DIR, fn))
    if sig.ndim > 1:
        sig = sig[:, 0]
    sig = sig.astype(np.float32)
    if sr_orig != SR:
        sig = librosa.resample(sig, orig_sr=sr_orig, target_sr=SR)

    sig_w = best_window(sig)
    feat  = build_features(sig_w)

    h = np.zeros((CFG.model.gru_layers, 1, CFG.model.gru_hidden), np.float32)
    outs = sess.run(None, {"features": feat, "h_in": h})
    kws_raw = outs[2][0]  # (4,)
    # apply softmax in case model outputs logits
    kws = np.exp(kws_raw - kws_raw.max())
    kws = kws / kws.sum()

    top = CLASSES[int(kws.argmax())]
    results.append(top)
    mark = "OK" if top == "stop" else "--"
    print("{:<32} {:>5.0f}% {:>5.0f}% {:>7.0f}% {:>5.0f}%  {} {}".format(
        fn, kws[0]*100, kws[1]*100, kws[2]*100, kws[3]*100, top.upper(), mark))

correct = results.count("stop")
print("\nResult: {}/{} = {:.0f}% classified as 'stop'".format(
    correct, len(results), 100 * correct / max(len(results), 1)))
print("Top-class distribution:", {c: results.count(c) for c in CLASSES})
