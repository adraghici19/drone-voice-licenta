"""
Adauga clipurile speech_real/ la manifeste (label 2 = speech).

Augmenteaza 5x (pitch ±1.5, ±3.0 + original) si le injecteaza
in train.csv/val.csv cu split 85/15.

Ruleaza dupa record_speech.py:
  python training/add_speech_to_manifest.py
Apoi:
  python training/finetune.py
"""

import os
import sys
import glob
import random
import numpy as np
import soundfile as sf
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SPEECH_DIR = "data/speech_real"
AUG_DIR    = "data/speech_real_aug"
TRAIN_CSV  = "data/manifests/train.csv"
VAL_CSV    = "data/manifests/val.csv"
KWS_LABEL  = 2   # speech
TRAIN_FRAC = 0.85
RANDOM_SEED = 42

try:
    import librosa
except ImportError:
    print("librosa necesar: pip install librosa"); sys.exit(1)

os.makedirs(AUG_DIR, exist_ok=True)

clips = sorted(glob.glob(os.path.join(SPEECH_DIR, "*.wav")))
if not clips:
    print(f"Nicio inregistrare gasita in {SPEECH_DIR}")
    print("Ruleaza mai intai: python pi_app/record_speech.py")
    sys.exit(1)

print(f"Gasit {len(clips)} clipuri originale in {SPEECH_DIR}")

# ── Augmentare ────────────────────────────────────────────────────────────────
PITCH_STEPS = [-3.0, -1.5, 0.0, +1.5, +3.0]
aug_paths   = []

for fpath in clips:
    y, sr = librosa.load(fpath, sr=None, mono=True)
    base  = os.path.splitext(os.path.basename(fpath))[0]

    for ps in PITCH_STEPS:
        if ps == 0.0:
            out_y = y.copy()
            tag   = "p+0.0"
        else:
            out_y = librosa.effects.pitch_shift(y, sr=sr, n_steps=ps)
            tag   = f"p{ps:+.1f}"

        peak = float(np.abs(out_y).max())
        if peak > 0.95:
            out_y = out_y * (0.95 / peak)

        fname = f"{base}_{tag}.wav"
        opath = os.path.join(AUG_DIR, fname)
        sf.write(opath, out_y.astype(np.float32), sr)
        dur = len(out_y) / sr
        aug_paths.append((opath, dur))

print(f"Augmentate: {len(aug_paths)} clipuri in {AUG_DIR}")

# ── Construieste randuri noi ───────────────────────────────────────────────────
random.seed(RANDOM_SEED)
random.shuffle(aug_paths)
split = int(len(aug_paths) * TRAIN_FRAC)
train_new = aug_paths[:split]
val_new   = aug_paths[split:]

def make_rows(paths, label):
    rows = []
    for p, dur in paths:
        rows.append({
            "speech_path": os.path.abspath(p),
            "kws_label":   label,
            "duration_s":  round(dur, 3),
            "is_voice":    1,
        })
    return rows

new_train_rows = make_rows(train_new, KWS_LABEL)
new_val_rows   = make_rows(val_new,   KWS_LABEL)

# ── Incarca si injecteaza in manifeste ────────────────────────────────────────
tr = pd.read_csv(TRAIN_CSV)
va = pd.read_csv(VAL_CSV)

# sterge orice speech_real deja existent (re-run safe)
tr = tr[~tr["speech_path"].str.contains("speech_real", na=False)]
va = va[~va["speech_path"].str.contains("speech_real", na=False)]

tr = pd.concat([tr, pd.DataFrame(new_train_rows)], ignore_index=True)
va = pd.concat([va, pd.DataFrame(new_val_rows)],   ignore_index=True)

tr.to_csv(TRAIN_CSV, index=False)
va.to_csv(VAL_CSV,   index=False)

# ── Raport ────────────────────────────────────────────────────────────────────
print("\n=== Distributie finala ===")
for csv_name, df in [("TRAIN", tr), ("VAL", va)]:
    print(f"\n{csv_name}:")
    for lbl, cnt in df["kws_label"].value_counts().sort_index().items():
        names = ["help", "stop", "speech", "noise"]
        name  = names[lbl] if lbl < len(names) else "?"
        print(f"  label {lbl} ({name:8s}): {cnt:4d} clips")

print(f"\nTrain +{len(new_train_rows)} / Val +{len(new_val_rows)} speech clips adaugate")
print("Acum ruleaza: python training/finetune.py")
