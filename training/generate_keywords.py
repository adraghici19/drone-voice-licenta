"""
Build clean, isolated 'help' / 'help_me' keyword clips and rebuild the manifests.

Why
---
The first hybrid run learned 'stop' (84 %) but completely failed 'help' /
'help_me' (0 %).  Root cause: those two classes were whole 5-15 s LibriSpeech
utterances containing the word somewhere, while the dataset crops a random
~0.53 s window — which almost never contains the keyword.  The label was noise.

This script replaces them with clean, isolated keyword clips (same recipe that
made 'stop' work via Google Speech Commands): SAPI-TTS base clips are
silence-trimmed so the word fills the clip, then augmented with pitch shifts and
time-stretch for variety.  The drone-noise + random-SNR mixing still happens at
training time inside ManifestDataset.

It then rebuilds train.csv / val.csv:
  * help / help_me  -> the new TTS clips (split train/val)
  * stop / background -> reused unchanged from the existing manifests

NOTE: TTS gives only two synthetic voices, so the model will learn the acoustic
pattern but generalisation to a real speaker is limited.  For the final thesis
demo, record your own 'help' / 'help me' clips and drop them into
data/tts_keywords/help{,_me}/ — this script will pick them up automatically.

Run:  python generate_keywords.py
Then: python run_training.py
"""

import os
import glob
import numpy as np
import soundfile as sf
import librosa
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TTS_DIR = os.path.join(HERE, "data", "tts_keywords")
MAN_DIR = os.path.join(HERE, "data", "manifests")

LABELS = {"help": 0, "help_me": 1}
PITCH_STEPS = [-3.0, -1.5, 0.0, 1.5, 3.0]   # semitones
TEMPO_RATES = [0.85, 1.0, 1.15]             # >1 = faster
VAL_FRACTION = 0.15
SEED = 7


def trim_and_pad(x, sr, target_s=0.6):
    """Trim leading/trailing silence; pad/crop to ~target_s so the word is centred."""
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x, _ = librosa.effects.trim(x, top_db=25)
    n_target = int(target_s * sr)
    if len(x) < n_target:                      # centre-pad
        pad = n_target - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    else:
        x = x[:n_target]
    peak = np.abs(x).max()
    if peak > 1e-6:
        x = x / peak * 0.95
    return x


def augment(x, sr):
    """Yield (suffix, signal) augmented variants."""
    for ps in PITCH_STEPS:
        xs = librosa.effects.pitch_shift(x, sr=sr, n_steps=ps) if ps != 0.0 else x
        for tr in TEMPO_RATES:
            xt = librosa.effects.time_stretch(xs, rate=tr) if tr != 1.0 else xs
            yield "p{:+.1f}_t{:.2f}".format(ps, tr), xt.astype(np.float32)


def build_clips():
    rng = np.random.default_rng(SEED)
    rows = []   # (path, label, duration_s)
    for name, label in LABELS.items():
        src_dir = os.path.join(TTS_DIR, name)
        out_dir = os.path.join(TTS_DIR, name + "_aug")
        os.makedirs(out_dir, exist_ok=True)
        # clean any previous run so re-running is idempotent
        for old in glob.glob(os.path.join(out_dir, "*.wav")):
            os.remove(old)

        bases = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
        if not bases:
            print("  WARNING: no base clips in", src_dir)
            continue
        n_made = 0
        for bp in bases:
            x, sr = sf.read(bp, dtype="float32")
            x = trim_and_pad(x, sr)
            stem = os.path.splitext(os.path.basename(bp))[0]
            for suffix, xa in augment(x, sr):
                op = os.path.join(out_dir, "{}_{}.wav".format(stem, suffix))
                sf.write(op, xa, sr)
                # 'group' = base recording, so all augmentations of one clip stay
                # together when splitting train/val (prevents data leakage)
                rows.append((op, label, len(xa) / sr, stem))
                n_made += 1
        print("  {:8s}: {} base -> {} augmented clips".format(name, len(bases), n_made))
    return pd.DataFrame(rows, columns=["speech_path", "kws_label", "duration_s", "group"]), rng


def rebuild_manifests(new_df, rng):
    train_csv = os.path.join(MAN_DIR, "train.csv")
    val_csv = os.path.join(MAN_DIR, "val.csv")
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)

    # keep stop (2) + background (3) exactly as they are (these already work)
    keep_cols = train.columns
    train_keep = train[train.kws_label.isin([2, 3])].copy()
    val_keep = val[val.kws_label.isin([2, 3])].copy()

    # split the new help/help_me clips into train/val at the BASE-RECORDING level
    # (all augmentations of one recording go to the same split — no leakage)
    new_train, new_val = [], []
    for label, grp in new_df.groupby("kws_label"):
        groups = grp["group"].unique()
        groups = rng.permutation(groups)
        n_val_groups = max(1, int(round(len(groups) * VAL_FRACTION)))
        val_groups = set(groups[:n_val_groups])
        new_val.append(grp[grp["group"].isin(val_groups)])
        new_train.append(grp[~grp["group"].isin(val_groups)])
    new_train = pd.concat(new_train, ignore_index=True).drop(columns=["group"])
    new_val = pd.concat(new_val, ignore_index=True).drop(columns=["group"])

    # add any missing columns (e.g. transcript) so schemas line up
    for col in keep_cols:
        if col not in new_train.columns:
            new_train[col] = ""
            new_val[col] = ""

    out_train = pd.concat([train_keep, new_train[keep_cols]], ignore_index=True)
    out_val = pd.concat([val_keep, new_val[keep_cols]], ignore_index=True)
    out_train = out_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out_val = out_val.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out_train.to_csv(train_csv, index=False)
    out_val.to_csv(val_csv, index=False)

    def dist(df, tag):
        c = df.kws_label.value_counts().sort_index()
        names = ["help", "help_me", "stop", "background"]
        print("  {}: total {}  ".format(tag, len(df)) +
              "  ".join("{}={}".format(names[i], int(c.get(i, 0))) for i in range(4)))

    print("Rebuilt manifests:")
    dist(out_train, "train")
    dist(out_val, "val")


if __name__ == "__main__":
    print("Generating clean keyword clips (TTS + augmentation) ...")
    df, rng = build_clips()
    print("Total new keyword clips:", len(df))
    rebuild_manifests(df, rng)
    print("DONE. Now run:  python run_training.py")
