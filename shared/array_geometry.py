import numpy as np

ARRAY_RADIUS_M: float = 0.045
SPEED_OF_SOUND_MS: float = 343.0

NUM_RAW_CHANNELS: int = 8
NUM_PHYSICAL_MICS: int = 7

_angles_deg = np.array([0.0, 0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
_radii       = np.array([0.0, ARRAY_RADIUS_M, ARRAY_RADIUS_M, ARRAY_RADIUS_M,
                          ARRAY_RADIUS_M, ARRAY_RADIUS_M, ARRAY_RADIUS_M])
_angles_rad = np.deg2rad(_angles_deg)

MIC_POSITIONS: np.ndarray = np.stack(
    [_radii * np.cos(_angles_rad),
     _radii * np.sin(_angles_rad)],
    axis=1,
)

NUM_ACTIVE_MICS: int = 6
ACTIVE_CHANNEL_INDICES: list[int] = [0, 1, 2, 3, 4, 6]

MIC_PAIRS: list[tuple[int, int]] = [(0, 1), (0, 3)]


def max_tdoa_seconds(pair: tuple[int, int]) -> float:
    i, j = pair
    pos_i = MIC_POSITIONS[ACTIVE_CHANNEL_INDICES[i]]
    pos_j = MIC_POSITIONS[ACTIVE_CHANNEL_INDICES[j]]
    return float(np.linalg.norm(pos_i - pos_j)) / SPEED_OF_SOUND_MS


def tdoa_to_phase(tdoa_s: float, freqs_hz: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * freqs_hz * tdoa_s
