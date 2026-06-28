import time
import numpy as np


class KeywordDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.classes = cfg.kws_classes
        self.bg = cfg.background_index
        self._ema = np.zeros(len(self.classes), dtype=np.float32)
        self._ema[self.bg] = 1.0
        self._last_fire = {i: 0.0 for i in range(len(self.classes))}

    def update(self, kws_probs: np.ndarray, now: float = None):
        if now is None:
            now = time.monotonic()
        a = self.cfg.kws_ema_alpha
        self._ema = a * kws_probs + (1.0 - a) * self._ema

        idx = int(self._ema.argmax())
        if idx == self.bg:
            return None
        if self._ema[idx] < self.cfg.kws_threshold:
            return None
        if now - self._last_fire[idx] < self.cfg.refractory_s:
            return None

        self._last_fire[idx] = now
        return self.classes[idx]

    @property
    def smoothed(self) -> np.ndarray:
        return self._ema
