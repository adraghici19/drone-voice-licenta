# pi_app — Real-time LiteVoiceNet inference (Raspberry Pi 4 + UMA-8 v2)

Streams audio from a **miniDSP UMA-8 v2** microphone array, runs the trained
**LiteVoiceNet** ONNX model on the CPU, and reports voice activity (VAD) and
detected keywords (`help`, `help_me`, `stop`).

The app is a thin, hardware-facing counterpart to the `training/` package and
has **no dependency on PyTorch** — only NumPy + ONNX Runtime at runtime.

## How it works

```
UMA-8 v2 (8 ch @ 48 kHz)
        │   (select logical mics 0,1,3)
        ▼
  sliding window  ~0.53 s (100 STFT frames)
        ▼
  build_features ── 1 log-magnitude + real IPD (sin/cos per mic pair) → (5, 100, 257)
        ▼
  ONNX Runtime  (INT8 preferred, FP32 fallback)
        ▼
  mask (denoise)  +  VAD (speech?)  +  KWS (4-class softmax)
        ▼
  KeywordDetector  ── EMA smoothing + threshold + refractory → keyword events
```

Each window is processed with a **fresh GRU state (zeros)** — the same regime
the model was trained under (every training sample was a ~0.53 s clip). The
window slides forward by `window_hop_frames` (≈133 ms) so keywords are caught
regardless of alignment.

### Why "real IPD", and why no beamforming (yet)

Training *simulated* the inter-channel phase difference (IPD) with a random
TDOA per clip and fed `sin/cos` of it. At deployment we compute the **true**
per-bin phase difference between the configured mic pairs —
`angle(STFT_a) − angle(STFT_b)` → `sin`, `cos`. This is the physically correct
counterpart of the simulated feature, and it is exactly the 5-channel layout
the network expects.

A GCC-PHAT DOA estimate + delay-and-sum beamformer (as sketched in the original
plan) is intentionally **not** used here: the current model was trained on
per-mic log-magnitude + IPD, not on a beamformed reference. Adding beamforming
would require retraining on beamformed features. It is a sensible future
enhancement, not a drop-in for this model.

## Files

| File | Purpose |
|------|---------|
| `config.py`   | Runtime constants — **must match** `training/config.py` signal/array settings |
| `features.py` | STFT + feature tensor builder (real IPD) |
| `audio.py`    | UMA-8 capture via `sounddevice` (ring buffer, channel selection) |
| `infer.py`    | ONNX Runtime session wrapper (INT8/FP32 auto-select) |
| `detect.py`   | KWS smoothing + threshold + refractory → discrete events |
| `main.py`     | Real-time loop and offline `--file` mode |
| `selftest.py` | Hardware-free pipeline validation |

## Install (on the Pi)

```bash
sudo apt update
sudo apt install libportaudio2 libsndfile1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# if onnxruntime has no wheel for your Python:
#   pip install --index-url https://www.piwheels.org/simple onnxruntime
```

Copy the exported models from the training machine, keeping the
`training/exported/` folder alongside `pi_app/`:

```
training/exported/litevoicenet.onnx        # FP32
training/exported/litevoicenet_int8.onnx   # INT8 (default on the Pi)
```

`infer.py` resolves these via the relative paths in `config.py`
(`onnx_int8` / `onnx_fp32`); edit those if you store the models elsewhere.

## Run

```bash
python selftest.py            # validate the pipeline (no mic required)
python main.py --list         # list audio devices → find the UMA-8 index
python main.py                # auto-detect UMA-8, INT8 model, start listening
python main.py --device 2     # force a device index
python main.py --fp32         # use the FP32 model
python main.py --file clip.wav  # offline run over a WAV (mono is broadcast to all mics)
```

Live output (one line per window):

```
[SPEECH] help=0.81  help_me=0.03  stop=0.05  background=0.11   >>> DETECTED: HELP <<<
[  --  ] help=0.04  help_me=0.01  stop=0.02  background=0.93
```

## Important: the model expects drone noise

LiteVoiceNet was trained exclusively on speech mixed with drone rotor noise. In
a **quiet room with no drone running**, clean speech is out-of-distribution and
keyword accuracy drops (≈75 % in offline tests). With realistic drone noise
present (the actual deployment condition), accuracy recovers to **≈96 %** on
held-out clips. Don't be alarmed by weaker numbers when testing indoors without
the drone — test with rotor noise (or a recording of it) for a representative
result.

The app defaults to the **FP32** model (`use_int8 = False`). FP32 is ~1.1 MB and
runs at ~24 ms/window (22× faster than real time), so INT8 buys nothing on the
Pi 4. The INT8 model is also provided and verified, but FP32 is the safe default.

## Configuration notes

- **UMA-8 channel mapping** — `config.py :: first_mic_channel`. With the stock
  DSP firmware, channel 0 is the processed output and raw mics start at 1
  (default). With raw-7-mic firmware, set it to `0`. The thesis noted channel 5
  was defective — confirm your mapping with a known tone source before relying
  on the IPD channels, and adjust `mic_pairs` if needed.
- **Detection tuning** — `kws_threshold` (fire sensitivity), `kws_ema_alpha`
  (smoothing), `refractory_s` (min gap between firings), `vad_threshold`.
- **CPU budget** — `intra_op_threads` (default 2 of the Pi's 4 cores). One-window
  CPU latency is reported by `training/run_training.py`'s benchmark; deploy the
  INT8 model.

## After retraining

`training/run_training.py` re-exports both ONNX files to `training/exported/`
at the end of every run, overwriting the old ones. No change needed here — the
Pi app picks up the new model on its next start.
