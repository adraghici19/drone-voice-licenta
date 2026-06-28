import numpy as np

C_SOUND = 343.0
ARRAY_RADIUS = 0.043


def hex_mic_positions(radius: float = ARRAY_RADIUS) -> np.ndarray:
    ang = np.deg2rad(np.arange(0, 360, 60))
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


class SrpPhatDoA:
    def __init__(self, sr: int, n_fft: int = 1024, mic_xy: np.ndarray = None,
                 grid_deg: float = 2.0, fmin: float = 300.0, fmax: float = 1700.0,
                 angle_offset_deg: float = 0.0):
        self.sr = sr
        self.n_fft = n_fft
        self.mic = hex_mic_positions() if mic_xy is None else np.asarray(mic_xy, float)
        self.M = len(self.mic)
        self.offset = angle_offset_deg

        self.grid = np.arange(0.0, 360.0, grid_deg)
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        self.band = (self.freqs >= fmin) & (self.freqs <= fmax)
        self.fb = self.freqs[self.band]

        self.pairs = [(i, j) for i in range(self.M) for j in range(i + 1, self.M)]

        dirs = np.stack([np.cos(np.deg2rad(self.grid)),
                         np.sin(np.deg2rad(self.grid))], axis=1)
        tau = -(dirs @ self.mic.T) / C_SOUND
        self._steer = np.zeros((len(self.grid), len(self.pairs), self.fb.size), np.complex128)
        for p, (i, j) in enumerate(self.pairs):
            dtau = tau[:, i] - tau[:, j]
            self._steer[:, p, :] = np.exp(-1j * 2 * np.pi * np.outer(dtau, self.fb))

    def estimate(self, frames: np.ndarray):
        win = np.hanning(self.n_fft)
        N = frames.shape[1]
        hop = self.n_fft // 2
        starts = range(0, max(N - self.n_fft, 0) + 1, hop)
        srp = np.zeros(len(self.grid))
        nwin = 0
        for s in starts:
            seg = frames[:, s:s + self.n_fft]
            if seg.shape[1] < self.n_fft:
                break
            spec = np.fft.rfft(seg * win, axis=1)[:, self.band]
            for p, (i, j) in enumerate(self.pairs):
                cross = spec[i] * np.conj(spec[j])
                cross /= (np.abs(cross) + 1e-9)
                srp += np.real(self._steer[:, p, :] @ np.conj(cross))
            nwin += 1
        if nwin == 0:
            return None, srp
        srp /= nwin
        az = float((self.grid[int(np.argmax(srp))] + self.offset) % 360.0)
        return az, srp


def _selftest():
    sr = 48000
    mic = hex_mic_positions()
    doa = SrpPhatDoA(sr, n_fft=1024, mic_xy=mic, grid_deg=2.0)
    rng = np.random.default_rng(0)
    n = int(sr * 0.3)
    t = np.arange(n) / sr
    src = np.zeros(n)
    for f0 in [300, 700, 1500, 2500]:
        src += np.sin(2 * np.pi * f0 * t) * rng.uniform(0.5, 1.0)
    src += 0.05 * rng.standard_normal(n)

    print("SRP-PHAT self-test:")
    errs = []
    for true_az in [0, 45, 90, 135, 200, 270, 315]:
        d = np.array([np.cos(np.deg2rad(true_az)), np.sin(np.deg2rad(true_az))])
        frames = np.zeros((len(mic), n))
        S = np.fft.rfft(src)
        f = np.fft.rfftfreq(n, 1 / sr)
        for m in range(len(mic)):
            tau = -(mic[m] @ d) / C_SOUND
            frames[m] = np.fft.irfft(S * np.exp(-1j * 2 * np.pi * f * tau), n=n)
        est, _ = doa.estimate(frames)
        err = min((est - true_az) % 360, (true_az - est) % 360)
        errs.append(err)
        print("  true {:3d}°  ->  est {:5.1f}°  (err {:.1f}°)".format(true_az, est, err))
    print("mean error: {:.1f}°".format(np.mean(errs)))


if __name__ == "__main__":
    _selftest()
