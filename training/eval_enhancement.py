import sys, os
sys.path.insert(0, '..'); sys.path.insert(0, '.')
import numpy as np, pandas as pd, soundfile as sf, librosa
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pystoi import stoi
from config import CFG
from data import _stft, _drone_noise, extract_features

sess = ort.InferenceSession('exported/litevoicenet.onnx', providers=['CPUExecutionProvider'])
h0 = np.zeros((CFG.model.gru_layers, 1, CFG.model.gru_hidden), np.float32)
sr = CFG.signal.sample_rate_raw
NF, HOP = CFG.signal.n_fft, CFG.signal.hop_size
n_samp = (CFG.synth.seq_frames - 1) * HOP + NF


def istft(S):
    win = np.hanning(NF)
    F, T = S.shape
    out = np.zeros((T - 1) * HOP + NF); w = np.zeros_like(out)
    for i in range(T):
        frame = np.fft.irfft(S[:, i], n=NF) * win
        out[i * HOP:i * HOP + NF] += frame
        w[i * HOP:i * HOP + NF] += win ** 2
    return out / (w + 1e-9)


def get_mask(mixture):
    peak = np.abs(mixture).max() + 1e-9
    mixn = mixture * (0.9 / peak)
    M = _stft(mixn, NF, HOP); F, T = M.shape
    freqs = np.fft.rfftfreq(NF, 1.0 / sr)
    log_mag = np.log(np.abs(M) ** 2 + 1e-8)
    rng = np.random.default_rng(0)
    chans = [log_mag]
    for _ in CFG.array.mic_pairs:
        tdoa = rng.uniform(-(2 * 0.045 / 343), (2 * 0.045 / 343))
        ipd = (2 * np.pi * freqs * tdoa)[:, None] * np.ones(T)
        chans += [np.sin(ipd).astype(np.float32), np.cos(ipd).astype(np.float32)]
    feats = np.stack(chans, 0).transpose(0, 2, 1).astype(np.float32)
    mask = sess.run(None, {'features': feats[None], 'h_in': h0})[0][0, 0]
    return mask.T


def snr(sig, ref):
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum((sig - ref) ** 2) + 1e-9) + 1e-9)


v = pd.read_csv('data/manifests/test.csv')
speech_clips = v[v.kws_label == 2].sample(30, random_state=0)
results = {}
demo = None
for snr_in in [0.0, 5.0, 10.0]:
    d_snr, d_stoi_in, d_stoi_out = [], [], []
    for _, r in speech_clips.iterrows():
        x, xs = sf.read(r.speech_path, dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        if xs != sr: x = librosa.resample(x, orig_sr=xs, target_sr=sr)
        x = x[:n_samp] if len(x) >= n_samp else np.pad(x, (0, n_samp - len(x)))
        rng = np.random.default_rng(int(snr_in) + len(d_snr))
        noise = _drone_noise(n_samp, sr, rng)
        s_rms = np.sqrt(np.mean(x ** 2) + 1e-9); n_rms = np.sqrt(np.mean(noise ** 2) + 1e-9)
        noise = noise * (s_rms / (n_rms * 10 ** (snr_in / 20)))
        mix = x + noise
        mask = get_mask(mix)
        Sx = _stft(x, NF, HOP); Sn = _stft(noise, NF, HOP)
        Tt = min(mask.shape[1], Sx.shape[1])
        sp_res = istft(Sx[:, :Tt] * mask[:, :Tt]); no_res = istft(Sn[:, :Tt] * mask[:, :Tt])
        snr_out = 10 * np.log10(np.sum(sp_res ** 2) / (np.sum(no_res ** 2) + 1e-9) + 1e-9)
        d_snr.append(snr_out - snr_in)
        enh = istft(_stft(mix, NF, HOP)[:, :Tt] * mask[:, :Tt])
        L = min(len(x), len(enh), len(mix))
        x16 = librosa.resample(x[:L], orig_sr=sr, target_sr=16000)
        try:
            d_stoi_in.append(stoi(x16, librosa.resample(mix[:L], orig_sr=sr, target_sr=16000), 16000, extended=False))
            d_stoi_out.append(stoi(x16, librosa.resample(enh[:L], orig_sr=sr, target_sr=16000), 16000, extended=False))
        except Exception:
            pass
        if demo is None and snr_in == 5.0:
            demo = (x, mix, enh, mask)
    results[snr_in] = (np.mean(d_snr), np.mean(d_stoi_in), np.mean(d_stoi_out))

print("Noise-cancelling (mask) evaluation on test speech:")
print("  input SNR | SNR improvement | STOI in -> STOI out")
for s, (dsnr, si, so) in results.items():
    print("   {:4.0f} dB  |   +{:4.1f} dB     |  {:.2f} -> {:.2f}".format(s, dsnr, si, so))

if demo is not None:
    x, mix, enh, mask = demo
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, sig, title in zip(axes, [x, mix, enh], ["Clean speech", "Noisy (speech+drone)", "Enhanced (mask applied)"]):
        Sg = np.log(np.abs(_stft(sig, NF, HOP)) ** 2 + 1e-8)
        ax.imshow(Sg, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title); ax.set_xlabel("time frame"); ax.set_yticks([])
    axes[0].set_ylabel("frequency")
    plt.tight_layout()
    out = os.path.normpath(os.path.join('..', 'docs', 'thesis', 'Figures', 'enhancement.png'))
    plt.savefig(out, dpi=130); plt.close()
    print("saved spectrogram figure ->", out)
