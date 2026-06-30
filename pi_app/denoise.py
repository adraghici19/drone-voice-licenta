import numpy as np


def _stft(x, n_fft, hop):
    win = np.hanning(n_fft)
    nfr = (len(x) - n_fft) // hop + 1
    out = np.empty((n_fft // 2 + 1, nfr), np.complex64)
    for i in range(nfr):
        out[:, i] = np.fft.rfft(x[i * hop:i * hop + n_fft] * win)
    return out


def _istft(S, n_fft, hop):
    win = np.hanning(n_fft)
    T = S.shape[1]
    out = np.zeros((T - 1) * hop + n_fft)
    wsum = np.zeros_like(out)
    for i in range(T):
        out[i * hop:i * hop + n_fft] += np.fft.irfft(S[:, i], n=n_fft) * win
        wsum[i * hop:i * hop + n_fft] += win ** 2
    return out / (wsum + 1e-9)


def estimate_noise_psd(power, percentile=15.0):
    return np.percentile(power, percentile, axis=1, keepdims=True)


def wiener_gain(power, noise_psd, oversub=1.5, gain_floor=0.1, alpha=0.95):
    noise_psd = np.maximum(noise_psd, 1e-10)
    snr_post = power / noise_psd
    gain = np.empty_like(power)
    prev = np.maximum(snr_post[:, :1] - 1.0, 1e-6)
    for t in range(power.shape[1]):
        sp = alpha * prev / (1 + prev) * snr_post[:, t:t+1] + \
             (1 - alpha) * np.maximum(snr_post[:, t:t+1] - oversub, 0.0)
        gain[:, t:t+1] = sp / (sp + 1.0)
        prev = sp
    return np.clip(gain, gain_floor, 1.0)


def denoise(noisy, n_fft=512, hop=256, oversub=1.5, gain_floor=0.1,
            alpha=0.95, noise_psd=None):
    S = _stft(noisy.astype(np.float64), n_fft, hop)
    power = np.abs(S) ** 2
    if noise_psd is None:
        noise_psd = estimate_noise_psd(power)
    gain = wiener_gain(power, noise_psd, oversub, gain_floor, alpha)
    return _istft(S * gain, n_fft, hop).astype(np.float32)
