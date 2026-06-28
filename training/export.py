import time
import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort


def export_onnx(
    model: nn.Module,
    cfg,
    output_path: str = "exported/litevoicenet.onnx",
    seq_frames: int = 100,
) -> str:
    model.eval()
    device = next(model.parameters()).device

    B, C, T, F = 1, cfg.array.num_features, seq_frames, cfg.signal.n_freq_bins
    dummy_x  = torch.zeros(B, C, T, F, device=device)
    dummy_hx = torch.zeros(cfg.model.gru_layers, B, cfg.model.gru_hidden, device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_x, dummy_hx),
        output_path,
        opset_version=cfg.onnx_opset,
        input_names=["features", "h_in"],
        output_names=["mask", "vad", "kws", "h_out"],
        dynamic_axes={
            "features": {0: "batch", 2: "time"},
            "h_in":     {1: "batch"},
            "mask":     {0: "batch", 2: "time"},
            "vad":      {0: "batch", 1: "time"},
            "kws":      {0: "batch"},
            "h_out":    {1: "batch"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(output_path) / 1e6
    print("ONNX export OK -> {}  ({:.1f} MB)".format(output_path, size_mb))
    return output_path


def export_tflite_int8(
    onnx_path: str,
    output_path: str = "exported/litevoicenet_int8.tflite",
    calib_data_provider: Optional[Callable] = None,
) -> Optional[str]:
    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        print(
            "\n[export_tflite_int8] onnx2tf not installed.\n"
            "  Run:  pip install onnx2tf tensorflow\n"
            "  Then re-run this cell."
        )
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tflite_dir = str(Path(output_path).parent / "tflite_tmp")

    try:
        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tflite_dir,
            output_integer_quantized_tflite=True,
            quant_type="per-tensor",
            verbosity="warn",
        )

        stem = Path(onnx_path).stem
        candidates = list(Path(tflite_dir).glob(f"{stem}*integer*.tflite"))
        if not candidates:
            candidates = list(Path(tflite_dir).glob("*.tflite"))
        if not candidates:
            print("[export_tflite_int8] No .tflite file found after conversion.")
            return None

        src = candidates[0]
        import shutil
        shutil.copy(src, output_path)
        size_mb = os.path.getsize(output_path) / 1e6
        print("TFLite INT8 export OK -> {}  ({:.1f} MB)".format(output_path, size_mb))
        return output_path

    except Exception as exc:
        print(f"[export_tflite_int8] Conversion failed: {exc}")
        return None


def quantize_onnx_int8(
    onnx_path: str,
    output_path: str = "exported/litevoicenet_int8.onnx",
    n_calib_samples: int = 200,
    cfg=None,
    calib_features: Optional[np.ndarray] = None,
) -> Optional[str]:
    try:
        from onnxruntime.quantization import (
            quantize_static, CalibrationDataReader, QuantType, QuantFormat,
        )
    except ImportError:
        print("[quantize_onnx_int8] onnxruntime quantisation not available.")
        return None

    if cfg is None:
        raise ValueError("cfg required for calibration data shape")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if calib_features is None:
        print("[quantize_onnx_int8] WARNING: no real calibration features given — "
              "using random data. The INT8 model will likely be broken. "
              "Pass calib_features from the dataset.")
        rng = np.random.default_rng(99)
        calib_features = rng.random(
            (n_calib_samples, cfg.array.num_features,
             cfg.synth.seq_frames, cfg.signal.n_freq_bins),
        ).astype(np.float32)
    else:
        calib_features = np.asarray(calib_features, dtype=np.float32)
        n_calib_samples = len(calib_features)

    class _Reader(CalibrationDataReader):
        def __init__(self, data, c_cfg):
            self._data = data
            self._i    = 0
            self._h = np.zeros(
                (c_cfg.model.gru_layers, 1, c_cfg.model.gru_hidden), dtype=np.float32
            )

        def get_next(self):
            if self._i >= len(self._data):
                return None
            feed = {
                "features": self._data[self._i : self._i + 1],
                "h_in":     self._h,
            }
            self._i += 1
            return feed

    try:
        reader = _Reader(calib_features, cfg)
        quantize_static(
            onnx_path,
            output_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        size_mb = os.path.getsize(output_path) / 1e6
        print("ONNX INT8 quantisation OK -> {}  ({:.1f} MB)".format(output_path, size_mb))
        return output_path
    except Exception as exc:
        print(f"[quantize_onnx_int8] Failed: {exc}")
        return None


def benchmark_cpu(
    onnx_path: str,
    cfg,
    n_runs: int = 200,
    n_warmup: int = 20,
) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(42)
    x  = rng.random((1, cfg.array.num_features, cfg.synth.seq_frames, cfg.signal.n_freq_bins)).astype(np.float32)
    hx = np.zeros((cfg.model.gru_layers, 1, cfg.model.gru_hidden), dtype=np.float32)
    feed = {"features": x, "h_in": hx}

    for _ in range(n_warmup):
        sess.run(None, feed)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    chunk_ms = cfg.synth.seq_frames * cfg.signal.hop_size / cfg.signal.sample_rate_raw * 1000
    stats = {
        "mean_ms":    float(arr.mean()),
        "p50_ms":     float(np.percentile(arr, 50)),
        "p95_ms":     float(np.percentile(arr, 95)),
        "chunk_audio_ms": chunk_ms,
        "realtime_factor": float(arr.mean() / chunk_ms),
    }
    return stats
