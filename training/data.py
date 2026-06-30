import os
import re
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt
from typing import List, Tuple, Optional


def _voiced_speech(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    f0 = rng.uniform(100.0, 250.0)
    t  = np.arange(n_samples) / sr

    sig = np.zeros(n_samples)
    for k in range(1, 9):
        sig += (1.0 / k) * np.sin(2.0 * np.pi * k * f0 * t)

    f_syl = rng.uniform(4.0, 8.0)
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * f_syl * t))
    sig *= envelope

    sos = butter(4, [80, 3500], btype="bandpass", fs=sr, output="sos")
    sig = sosfilt(sos, sig)

    peak = np.abs(sig).max()
    return sig / (peak + 1e-9)


def _drone_noise(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    f_rotor = rng.uniform(80.0, 150.0)
    t = np.arange(n_samples) / sr

    sig = np.zeros(n_samples)
    for k in range(1, 14):
        amp = rng.uniform(0.4, 1.0) / k
        sig += amp * np.sin(2.0 * np.pi * k * f_rotor * t + rng.uniform(0, 2 * np.pi))

    broadband = rng.standard_normal(n_samples)
    sos = butter(4, 2000, btype="lowpass", fs=sr, output="sos")
    broadband = sosfilt(sos, broadband) * 0.4
    sig += broadband

    peak = np.abs(sig).max()
    return sig / (peak + 1e-9)


def _add_fullband(signal: np.ndarray, rng: np.random.Generator,
                  augment: bool = True) -> np.ndarray:
    n = len(signal)
    white = rng.standard_normal(n).astype(np.float32)
    s_rms = np.sqrt(np.mean(signal ** 2) + 1e-9)
    w_rms = np.sqrt(np.mean(white ** 2) + 1e-9)
    snr = rng.uniform(15.0, 40.0) if augment else 25.0
    return signal + white * (s_rms / (w_rms * 10 ** (snr / 20.0)))


def _stft(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft)
    n_frames = (len(signal) - n_fft) // hop + 1
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.complex64)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = signal[idx] * window
    return np.fft.rfft(frames, axis=1).T.astype(np.complex64)


def extract_features(
    speech: np.ndarray,
    noise:  np.ndarray,
    snr_db: float,
    mic_pairs: List[Tuple[int, int]],
    n_fft: int,
    hop: int,
    sr: int,
    rng: np.random.Generator,
    array_radius_m: float = 0.045,
    c: float = 343.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(speech)

    s_rms = np.sqrt(np.mean(speech ** 2) + 1e-9)
    n_rms = np.sqrt(np.mean(noise  ** 2) + 1e-9)
    noise_scaled = noise * (s_rms / (n_rms * 10 ** (snr_db / 20)))

    mixture = speech + noise_scaled

    peak = np.abs(mixture).max()
    scale = 0.9 / (peak + 1e-9)
    mixture      = mixture      * scale
    speech_norm  = speech       * scale
    noise_norm   = noise_scaled * scale

    S = _stft(speech_norm, n_fft, hop)
    N = _stft(noise_norm,  n_fft, hop)
    M = _stft(mixture,     n_fft, hop)

    F_bins, T_frames = S.shape
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    log_mag = np.log(np.abs(M) ** 2 + 1e-8)

    s_pow = np.abs(S) ** 2
    n_pow = np.abs(N) ** 2
    irm   = s_pow / (s_pow + n_pow + 1e-9)

    frame_e = s_pow.mean(axis=0)
    vad = (frame_e > 0.1 * frame_e.max()).astype(np.float32)

    feature_channels = [log_mag]
    max_tdoa = (2 * array_radius_m) / c

    for _ in mic_pairs:
        tdoa = rng.uniform(-max_tdoa, max_tdoa)
        phase_delay = 2.0 * np.pi * freqs * tdoa
        ipd = phase_delay[:, np.newaxis] * np.ones(T_frames)
        feature_channels.append(np.sin(ipd).astype(np.float32))
        feature_channels.append(np.cos(ipd).astype(np.float32))

    features = np.stack(feature_channels, axis=0).transpose(0, 2, 1)

    return (
        features.astype(np.float32),
        irm.T.astype(np.float32),
        vad,
    )


def extract_features_noise(
    mixture: np.ndarray,
    mic_pairs: List[Tuple[int, int]],
    n_fft: int,
    hop: int,
    sr: int,
    rng: np.random.Generator,
    array_radius_m: float = 0.045,
    c: float = 343.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak = np.abs(mixture).max()
    mixture = mixture * (0.9 / (peak + 1e-9))

    M = _stft(mixture, n_fft, hop)
    F_bins, T_frames = M.shape
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    log_mag = np.log(np.abs(M) ** 2 + 1e-8)
    feature_channels = [log_mag]
    max_tdoa = (2 * array_radius_m) / c
    for _ in mic_pairs:
        tdoa = rng.uniform(-max_tdoa, max_tdoa)
        phase_delay = 2.0 * np.pi * freqs * tdoa
        ipd = phase_delay[:, np.newaxis] * np.ones(T_frames)
        feature_channels.append(np.sin(ipd).astype(np.float32))
        feature_channels.append(np.cos(ipd).astype(np.float32))

    features = np.stack(feature_channels, axis=0).transpose(0, 2, 1)
    irm = np.zeros((T_frames, F_bins), dtype=np.float32)
    vad = np.zeros(T_frames, dtype=np.float32)
    return features.astype(np.float32), irm, vad


class SyntheticAudioDataset:

    def __init__(self, num_samples: int, cfg, base_seed: int = 0):
        self.num_samples = num_samples
        self.cfg         = cfg
        self.base_seed   = base_seed

        self.sr        = cfg.signal.sample_rate_raw
        self.n_fft     = cfg.signal.n_fft
        self.hop       = cfg.signal.hop_size
        self.n_frames  = cfg.synth.seq_frames
        self.snr_min   = cfg.synth.snr_min_db
        self.snr_max   = cfg.synth.snr_max_db
        self.mic_pairs = cfg.array.mic_pairs
        self.n_kws     = cfg.model.num_kws_classes

        self.n_samples = (self.n_frames - 1) * self.hop + self.n_fft

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + idx)

        speech = _voiced_speech(self.n_samples, self.sr, rng)
        noise  = _drone_noise(self.n_samples, self.sr, rng)
        snr_db = rng.uniform(self.snr_min, self.snr_max)
        kws_label = int(rng.integers(0, self.n_kws))

        features, irm, vad = extract_features(
            speech, noise, snr_db, self.mic_pairs,
            self.n_fft, self.hop, self.sr, rng,
        )

        return features, irm, vad, int(kws_label)


def build_datasets(cfg):
    train_csv = os.path.join(cfg.data_dir, 'manifests', 'train.csv')
    val_csv   = os.path.join(cfg.data_dir, 'manifests', 'val.csv')

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        print("ManifestDataset  ->  train: {}  val: {}".format(train_csv, val_csv))
        return (ManifestDataset(train_csv, cfg, augment=True),
                ManifestDataset(val_csv,   cfg, augment=False))

    print("Manifests not found -> SyntheticAudioDataset  "
          "(run download_data.py for real audio)")
    return (SyntheticAudioDataset(cfg.synth.train_samples, cfg,
                                   base_seed=cfg.synth.seed),
            SyntheticAudioDataset(cfg.synth.val_samples,   cfg,
                                   base_seed=cfg.synth.seed + 100_000))


def _spec_augment(features: np.ndarray, rng: np.random.Generator,
                  n_time: int = 2, max_t: int = 12,
                  n_freq: int = 2, max_f: int = 20) -> np.ndarray:
    C, T, F = features.shape
    features = features.copy()
    for _ in range(n_time):
        if rng.random() < 0.5:
            w = int(rng.integers(1, max_t + 1))
            s = int(rng.integers(0, max(T - w, 1)))
            features[:, s:s + w, :] = 0.0
    log_mean = float(features[0].mean())
    for _ in range(n_freq):
        if rng.random() < 0.5:
            w = int(rng.integers(1, max_f + 1))
            s = int(rng.integers(0, max(F - w, 1)))
            features[0, :, s:s + w] = log_mean
    return features


class ManifestDataset:

    def __init__(self, manifest_path: str, cfg, augment: bool = False,
                 base_seed: int = 5000):
        import pandas as pd
        self.df        = pd.read_csv(manifest_path)
        self.cfg       = cfg
        self.augment   = augment
        self.base_seed = base_seed

        self.sr        = cfg.signal.sample_rate_raw
        self.n_fft     = cfg.signal.n_fft
        self.hop       = cfg.signal.hop_size
        self.n_frames  = cfg.synth.seq_frames
        self.snr_min   = cfg.synth.snr_min_db
        self.snr_max   = cfg.synth.snr_max_db
        self.snr_mid   = (cfg.synth.snr_min_db + cfg.synth.snr_max_db) / 2
        self.mic_pairs = cfg.array.mic_pairs

        self.n_samples = (self.n_frames - 1) * self.hop + self.n_fft

        self.bg_pool = (self.df[self.df['is_voice'] == 0]['speech_path'].tolist()
                        if 'is_voice' in self.df.columns else [])

        self._cache_dir = Path(manifest_path).parent / '.resample_cache'
        self._cache_dir.mkdir(exist_ok=True)
        self._precache()

    def _cache_path(self, speech_path: str) -> Path:
        import hashlib
        key = hashlib.md5((str(speech_path) + str(self.sr)).encode()).hexdigest()
        return self._cache_dir / (key + '.npy')

    def _precache(self):
        import soundfile as sf
        import librosa

        paths   = self.df['speech_path'].tolist()
        missing = [p for p in paths if not self._cache_path(p).exists()]

        if not missing:
            print("  Resample cache: all {} files ready.".format(len(paths)))
            return

        print("  Resampling {} / {} files to cache (first run) ...".format(
            len(missing), len(paths)))
        t0 = __import__('time').time()
        for i, path in enumerate(missing):
            if i % 1000 == 0 and i > 0:
                elapsed = __import__('time').time() - t0
                eta = elapsed / i * (len(missing) - i)
                print("    {}/{} ({:.0f}s elapsed, ETA {:.0f}s)".format(
                    i, len(missing), elapsed, eta))
            try:
                info = sf.info(str(path))
                nframes = min(info.frames, 15 * info.samplerate)
                speech, sr_file = sf.read(str(path), frames=nframes,
                                          dtype='float32', always_2d=False)
                if speech.ndim > 1:
                    speech = speech.mean(axis=1)
                if sr_file != self.sr:
                    speech = librosa.resample(speech, orig_sr=sr_file,
                                               target_sr=self.sr)
                np.save(self._cache_path(path), speech)
            except Exception as e:
                np.save(self._cache_path(path), np.zeros(self.n_samples, dtype=np.float32))

        elapsed = __import__('time').time() - t0
        print("  Pre-cache done in {:.1f} s".format(elapsed))

    def _load_speech(self, speech_path: str) -> np.ndarray:
        cp = self._cache_path(speech_path)
        if cp.exists():
            return np.load(cp)
        import soundfile as sf, librosa
        speech, sr_file = sf.read(str(speech_path), dtype='float32', always_2d=False)
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if sr_file != self.sr:
            speech = librosa.resample(speech, orig_sr=sr_file, target_sr=self.sr)
        return speech

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + idx)
        row = self.df.iloc[idx]

        speech = self._load_speech(str(row.speech_path))

        n = len(speech)
        if n >= self.n_samples:
            if self.augment and rng.random() < 0.15:
                # Occasionally crop a short burst + silence from long clips so the
                # model can't use "voice burst followed by padding" alone as a
                # shortcut for the keyword classes (help/stop are naturally short).
                max_short = max(self.n_samples // 2, 1)
                short_n = min(int(rng.uniform(0.3, 0.7) * self.sr), max_short, n)
                short_n = max(short_n, 1)
                st = int(rng.integers(0, n - short_n + 1))
                clip = speech[st : st + short_n]
                pad_total = self.n_samples - short_n
                pad_before = int(rng.integers(0, pad_total + 1))
                speech = np.pad(clip, (pad_before, pad_total - pad_before))
            elif self.augment:
                start = int(rng.integers(0, n - self.n_samples + 1))
                speech = speech[start : start + self.n_samples]
            else:
                speech = speech[0 : self.n_samples]
        else:
            pad_total = self.n_samples - n
            pad_before = int(rng.integers(0, pad_total + 1)) if self.augment else 0
            speech = np.pad(speech, (pad_before, pad_total - pad_before))

        is_voice = bool(row['is_voice']) if 'is_voice' in self.df.columns else True

        if is_voice:
            speech = _add_fullband(speech, rng, self.augment)
            noise = _drone_noise(self.n_samples, self.sr, rng)
            if self.augment and self.bg_pool and rng.random() < 0.35:
                bg = self._load_speech(self.bg_pool[int(rng.integers(len(self.bg_pool)))])
                if len(bg) >= self.n_samples:
                    st = int(rng.integers(0, len(bg) - self.n_samples + 1))
                    bg = bg[st:st + self.n_samples]
                else:
                    bg = np.pad(bg, (0, self.n_samples - len(bg)))
                d_rms = np.sqrt(np.mean(noise ** 2) + 1e-9)
                b_rms = np.sqrt(np.mean(bg ** 2) + 1e-9)
                noise = noise + bg * (d_rms / b_rms) * float(rng.uniform(0.2, 0.8))
            snr_db = (float(rng.uniform(self.snr_min, self.snr_max))
                      if self.augment else self.snr_mid)
            features, irm, vad = extract_features(
                speech, noise, snr_db, self.mic_pairs,
                self.n_fft, self.hop, self.sr, rng,
            )
        else:
            drone = _drone_noise(self.n_samples, self.sr, rng)
            e_rms = np.sqrt(np.mean(speech ** 2) + 1e-9)
            d_rms = np.sqrt(np.mean(drone ** 2) + 1e-9)
            g = float(rng.uniform(0.2, 1.0)) if self.augment else 0.5
            mixture = _add_fullband(speech + g * drone * (e_rms / d_rms), rng, self.augment)
            features, irm, vad = extract_features_noise(
                mixture, self.mic_pairs, self.n_fft, self.hop, self.sr, rng,
            )

        if self.augment:
            features = _spec_augment(features, rng)

        return features, irm, vad, int(row.kws_label)
