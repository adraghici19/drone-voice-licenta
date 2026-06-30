from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SignalConfig:
    sample_rate_raw: int = 48_000
    sample_rate_kws: int = 16_000
    n_fft: int = 512
    hop_size: int = 256
    n_freq_bins: int = 257


@dataclass
class ArrayConfig:
    num_active_mics: int = 6
    mic_pairs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 1), (0, 3)]
    )

    @property
    def num_features(self) -> int:
        return 1 + 2 * len(self.mic_pairs)


@dataclass
class ModelConfig:
    enc_ch: List[int] = field(default_factory=lambda: [32, 64, 96, 128])
    gru_hidden: int = 128
    gru_layers: int = 2
    num_kws_classes: int = 4
    kws_classes: List[str] = field(
        default_factory=lambda: ["help", "stop", "speech", "noise"]
    )
    kws_window_frames: int = 188


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 6e-4
    num_epochs: int = 30
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class LossConfig:
    w_mask: float = 1.0
    w_vad: float = 0.5
    w_kws: float = 0.5


@dataclass
class SynthConfig:
    train_samples: int = 2000
    val_samples: int = 400
    seq_frames: int = 100
    snr_min_db: float = -2.0
    snr_max_db: float = 15.0
    seed: int = 0


@dataclass
class Config:
    signal: SignalConfig = field(default_factory=SignalConfig)
    array: ArrayConfig = field(default_factory=ArrayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)

    export_dir: str = "exported"
    data_dir:   str = "data"
    onnx_opset: int = 17


CFG = Config()
