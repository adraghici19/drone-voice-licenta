"""
Augments stop_real clips (5x: pitch +-1.5, +-3.0 semitones + original)
and adds them to train.csv / val.csv as label 1 (stop class).
"""

import os, sys
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

STOP_DIR  = os.path.join("data", "stop_real")
AUG_DIR   = os.path.join("data", "stop_real_aug")
TRAIN_CSV = os.path.join("data", "manifests", "train.csv")
VAL_CSV   = os.path.join("data", "manifests", "val.csv")
LABEL     = 1  # stop class

PITCH_SHIFTS = [-3.0, -1.5, 0.0, 1.5, 3.0]

os.makedirs(AUG_DIR, exist_ok=True)

wavs = sorted(f for f in os.listdir(STOP_DIR) if f.endswith(".wav"))
print("Found {} stop_real clips".format(len(wavs)))

aug_paths = []
for fn in wavs:
    src = os.path.join(STOP_DIR, fn)
    sig, sr = sf.read(src)
    base = os.path.splitext(fn)[0]
    for shift in PITCH_SHIFTS:
        shifted = librosa.effects.pitch_shift(sig.astype(np.float32), sr=sr, n_steps=shift)
        tag = "orig" if shift == 0.0 else "p{:+.1f}".format(shift).replace("+", "p").replace("-", "m").replace(".", "d")
        out_fn = "{}_{}.wav".format(base, tag)
        out_path = os.path.join(AUG_DIR, out_fn)
        sf.write(out_path, shifted, sr)
        aug_paths.append(os.path.abspath(out_path))

print("Generated {} augmented clips".format(len(aug_paths)))

# 85/15 train/val split
train_p, val_p = train_test_split(aug_paths, test_size=0.15, random_state=42)
print("Split: {} train / {} val".format(len(train_p), len(val_p)))

def append_to_csv(csv_path, paths, label):
    df_exist = pd.read_csv(csv_path)
    new_rows = pd.DataFrame({"path": paths, "kws_label": label})
    df_new = pd.concat([df_exist, new_rows], ignore_index=True)
    df_new.to_csv(csv_path, index=False)
    return len(df_new)

n_train = append_to_csv(TRAIN_CSV, train_p, LABEL)
n_val   = append_to_csv(VAL_CSV,   val_p,   LABEL)
print("train.csv now has {} rows".format(n_train))
print("val.csv   now has {} rows".format(n_val))
print("\nNext: .venv\\Scripts\\python.exe training\\finetune.py")
