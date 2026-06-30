import sys
import numpy as np

try:
    import sounddevice as sd
except Exception as exc:
    sd = None
    _IMPORT_ERR = exc


def list_devices():
    if sd is None:
        print("sounddevice not available:", _IMPORT_ERR)
        return
    print(sd.query_devices())


def find_uma8(name_hint: str = "UMA") -> int:
    if sd is None:
        raise RuntimeError("sounddevice not available: {}".format(_IMPORT_ERR))
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] >= 1 and name_hint.lower() in dev["name"].lower():
            return idx
    raise RuntimeError(
        "UMA-8 not found. Run audio.list_devices() to see options."
    )


class MicArrayStream:
    def __init__(self, cfg, device=None):
        if sd is None:
            raise RuntimeError("sounddevice not available: {}".format(_IMPORT_ERR))
        self.cfg = cfg
        self.device = device

        self.mic_channels = [
            cfg.first_mic_channel + i
            for i in sorted({cfg.ref_mic} | {m for p in cfg.mic_pairs for m in p})
        ]
        self.logical_index = {
            mic: row for row, mic in enumerate(
                sorted({cfg.ref_mic} | {m for p in cfg.mic_pairs for m in p})
            )
        }

        self._ring_len = cfg.window_samples * 3
        self._ring = np.zeros((cfg.num_device_channels, self._ring_len), dtype=np.float32)
        self._write = 0
        self._filled = 0

        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.num_device_channels,
            dtype="float32",
            device=device,
            blocksize=cfg.hop_size,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("[audio] status:", status, file=sys.stderr)
        data = indata.T
        end = self._write + frames
        if end <= self._ring_len:
            self._ring[:, self._write:end] = data
        else:
            first = self._ring_len - self._write
            self._ring[:, self._write:] = data[:, :first]
            self._ring[:, : frames - first] = data[:, first:]
        self._write = end % self._ring_len
        self._filled = min(self._filled + frames, self._ring_len)

    def _latest(self, n_samples: int) -> np.ndarray:
        idx = (self._write - n_samples) % self._ring_len
        if idx + n_samples <= self._ring_len:
            return self._ring[:, idx: idx + n_samples].copy()
        first = self._ring_len - idx
        out = np.empty((self.cfg.num_device_channels, n_samples), dtype=np.float32)
        out[:, :first] = self._ring[:, idx:]
        out[:, first:] = self._ring[:, : n_samples - first]
        return out

    def has_window(self) -> bool:
        return self._filled >= self.cfg.window_samples

    def read_window(self) -> np.ndarray:
        raw = self._latest(self.cfg.window_samples)
        rows = max(self.logical_index) + 1
        out = np.zeros((rows, self.cfg.window_samples), dtype=np.float32)
        for logical, row in self.logical_index.items():
            out[logical] = raw[self.cfg.first_mic_channel + logical]
        return out

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()
