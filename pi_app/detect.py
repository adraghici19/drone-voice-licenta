import time
import numpy as np


class KeywordDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.classes = cfg.kws_classes
        self.bg = cfg.background_index
        self._ema = np.zeros(len(self.classes), dtype=np.float32)
        self._ema[self.bg] = 1.0
        self._last_fire = 0.0
        self._consec = 0
        self._consec_idx = -1

    def update(self, kws_probs: np.ndarray, now: float = None):
        if now is None:
            now = time.monotonic()
        a = self.cfg.kws_ema_alpha
        self._ema = a * kws_probs + (1.0 - a) * self._ema

        idx = int(self._ema.argmax())
        thresholds = getattr(self.cfg, 'kws_class_thresholds', None)
        threshold = thresholds[idx] if thresholds else self.cfg.kws_threshold
        if idx == self.bg or self._ema[idx] < threshold:
            self._consec = 0
            self._consec_idx = -1
            return None

        if idx == self._consec_idx:
            self._consec += 1
        else:
            self._consec = 1
            self._consec_idx = idx

        if self._consec < self.cfg.kws_min_frames:
            return None
        if now - self._last_fire < self.cfg.refractory_s:
            return None

        self._last_fire = now
        self._consec = 0
        return self.classes[idx]

    @property
    def smoothed(self) -> np.ndarray:
        return self._ema
