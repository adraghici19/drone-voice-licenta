import numpy as np
from typing import List, Tuple

_EPS = 1e-9


def stft(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft)
    n_frames = (len(signal) - n_fft) // hop + 1
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.complex64)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = signal[idx] * window
    return np.fft.rfft(frames, axis=1).T.astype(np.complex64)


def build_features(
    channels: np.ndarray,
    mic_pairs: List[Tuple[int, int]],
    ref_mic: int,
    n_fft: int,
    hop: int,
) -> np.ndarray:
    peak = np.abs(channels[ref_mic]).max()
    channels = channels * (0.9 / (peak + _EPS))

    needed = sorted({ref_mic} | {m for pair in mic_pairs for m in pair})
    specs = {m: stft(channels[m], n_fft, hop) for m in needed}

    M_ref = specs[ref_mic]
    log_mag = np.log(np.abs(M_ref) ** 2 + 1e-8)

    feature_channels = [log_mag]
    for a, b in mic_pairs:
        ipd = np.angle(specs[a]) - np.angle(specs[b])
        feature_channels.append(np.sin(ipd).astype(np.float32))
        feature_channels.append(np.cos(ipd).astype(np.float32))

    features = np.stack(feature_channels, axis=0).transpose(0, 2, 1)
    return features.astype(np.float32)
