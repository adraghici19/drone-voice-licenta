import sys, os, importlib.util
sys.path.insert(0, '..'); sys.path.insert(0, '.')
import numpy as np, pandas as pd, soundfile as sf, librosa
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pystoi import stoi
from config import CFG
from data import _drone_noise

spec = importlib.util.spec_from_file_location("dn", os.path.abspath("../pi_app/denoise.py"))
dn = importlib.util.module_from_spec(spec); spec.loader.exec_module(dn)

sr = CFG.signal.sample_rate_raw
NF, HOP = CFG.signal.n_fft, CFG.signal.hop_size
DUR = 3.0
N = int(DUR * sr)

v = pd.read_csv('data/manifests/test.csv')
clips = v[v.kws_label == 2].sample(30, random_state=0)
demo = None
print("Classical Wiener noise-cancelling on drone noise:")
print("  in SNR | SNR improvement | STOI in -> out")
for snr_in in [0.0, 5.0, 10.0]:
    dsnr, si, so = [], [], []
    for _, r in clips.iterrows():
        x, xs = sf.read(r.speech_path, dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        if xs != sr: x = librosa.resample(x, orig_sr=xs, target_sr=sr)
        if len(x) < N: x = np.pad(x, (0, N - len(x)))
        else: x = x[:N]
        rng = np.random.default_rng(int(snr_in) + len(dsnr))
        noise = _drone_noise(N, sr, rng)
        s_rms = np.sqrt(np.mean(x**2)+1e-9); n_rms = np.sqrt(np.mean(noise**2)+1e-9)
        noise = noise * (s_rms / (n_rms * 10**(snr_in/20)))
        mix = x + noise
        Smix = dn._stft(mix, NF, HOP); Sx = dn._stft(x, NF, HOP); Sn = dn._stft(noise, NF, HOP)
        gain = dn.wiener_gain(np.abs(Smix)**2, dn.estimate_noise_psd(np.abs(Smix)**2))
        sp_res = dn._istft(Sx * gain, NF, HOP); no_res = dn._istft(Sn * gain, NF, HOP)
        out_snr = 10*np.log10(np.sum(sp_res**2)/(np.sum(no_res**2)+1e-9)+1e-9)
        dsnr.append(out_snr - snr_in)
        enh = dn._istft(Smix * gain, NF, HOP)
        L = min(len(x), len(enh), len(mix))
        x16 = librosa.resample(x[:L], orig_sr=sr, target_sr=16000)
        try:
            si.append(stoi(x16, librosa.resample(mix[:L], orig_sr=sr, target_sr=16000), 16000))
            so.append(stoi(x16, librosa.resample(enh[:L], orig_sr=sr, target_sr=16000), 16000))
        except Exception:
            pass
        if demo is None and snr_in == 5.0:
            demo = (x, mix, enh)
    print("   {:4.0f}  |   +{:4.1f} dB     |  {:.2f} -> {:.2f}".format(snr_in, np.mean(dsnr), np.mean(si), np.mean(so)))

if demo:
    x, mix, enh = demo
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, sig, t in zip(axes, [x, mix, enh], ["Clean speech", "Noisy (+ drone)", "Denoised (Wiener)"]):
        Sg = np.log(np.abs(dn._stft(sig, NF, HOP))**2 + 1e-8)
        ax.imshow(Sg, origin='lower', aspect='auto', cmap='magma'); ax.set_title(t); ax.set_xlabel('frame'); ax.set_yticks([])
    axes[0].set_ylabel('frequency')
    plt.tight_layout()
    out = os.path.normpath(os.path.join('..', 'docs', 'thesis', 'Figures', 'enhancement.png'))
    plt.savefig(out, dpi=130); plt.close()
    print("saved spectrogram ->", out)
