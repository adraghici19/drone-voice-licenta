from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PiConfig:
    sample_rate: int = 48_000
    n_fft: int = 512
    hop_size: int = 256
    n_freq_bins: int = 257

    seq_frames: int = 100
    window_hop_frames: int = 25

    num_device_channels: int = 8
    first_mic_channel: int = 0
    mic_pairs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 1), (0, 3)]
    )
    ref_mic: int = 0

    @property
    def num_features(self) -> int:
        return 1 + 2 * len(self.mic_pairs)

    @property
    def window_samples(self) -> int:
        return (self.seq_frames - 1) * self.hop_size + self.n_fft

    @property
    def hop_samples(self) -> int:
        return self.window_hop_frames * self.hop_size

    gru_layers: int = 2
    gru_hidden: int = 128
    kws_classes: List[str] = field(
        default_factory=lambda: ["help", "stop", "speech", "noise"]
    )
    keyword_indices: List[int] = field(default_factory=lambda: [0, 1])
    background_index: int = 3

    kws_ema_alpha: float = 0.4
    kws_threshold: float = 0.6
    refractory_s: float = 1.0
    vad_threshold: float = 0.5

    onnx_fp32: str = "../training/exported/litevoicenet.onnx"
    onnx_int8: str = "../training/exported/litevoicenet_int8.onnx"
    use_int8: bool = False

    intra_op_threads: int = 2


PI_CFG = PiConfig()
