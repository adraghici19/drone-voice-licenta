"""
Step 2 — Download LibriSpeech + Google Speech Commands and create manifest CSVs.

Downloads:
    dev-clean          337 MB    (val speech, LibriSpeech)
    train-clean-100      6.3 GB  (train speech, LibriSpeech)
    speech_commands_v0.02  2.3 GB  (isolated 1-s keyword clips)

Creates:
    data/manifests/train.csv
    data/manifests/val.csv

Manifest columns: speech_path, kws_label, duration_s, transcript

KWS label assignment:
    0  help        LibriSpeech utterances containing "help" (short utterances)
    1  help_me     LibriSpeech utterances containing "help me" (short utterances)
    2  stop        Google Speech Commands "stop" class (perfect 1-s clips)
    3  background  LibriSpeech non-keyword + Speech Commands non-stop words

Usage:
    python download_data.py              # full hybrid dataset (recommended)
    python download_data.py --dev-only   # dev-clean only LibriSpeech, no Speech Commands
    python download_data.py --check      # show what exists, no download
"""

import argparse
import csv
import os
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

# Config

LIBRI_BASE_URL = "https://www.openslr.org/resources/12/"
LIBRI_ARCHIVES = {
    "dev-clean":        ("dev-clean.tar.gz",        337),
    "train-clean-100":  ("train-clean-100.tar.gz",  6_300),
}

SC_URL      = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
SC_SIZE_MB  = 2_300
SC_DIRNAME  = "speech_commands_v0.02"   # preferred subdir; falls back to flat extraction

# Speech Commands words mapped to KWS labels (only "stop" is a keyword)
SC_KEYWORD_MAP = {"stop": 2}

# Words used as high-quality background hard-negatives (isolated clean speech)
SC_BACKGROUND_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
]

KWS_CLASSES = ["help", "help_me", "stop", "background"]

# Max utterance duration for LibriSpeech keyword classes.
# LibriSpeech sentences are typically 5-15 s; "help" appears in long sentences,
# so we keep all utterances (None = no filter).  With random cropping over 30
# epochs, a 10-s utterance still has ~5 % chance per crop of capturing the
# keyword; across 30 epochs that is >78 % probability of at least one hit.
MAX_KEYWORD_DURATION_S = None  # None = keep all

# Background caps (relative to stop count to keep training balanced)
BG_SC_MULTIPLIER   = 2   # non-stop Speech Commands words: 2× stop count
BG_LIBRI_EXTRA     = 500  # additional LibriSpeech background on top of SC bg


# Progress bar

def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb  = downloaded / 1e6
        tot = total_size / 1e6
        bar = '#' * int(pct / 2)
        sys.stdout.write(
            "\r  [{:<50}] {:.1f}/{:.1f} MB  {:.1f}%".format(bar, mb, tot, pct)
        )
        sys.stdout.flush()


# Download + extract

def download_file(url: str, dest_path: Path, size_mb: int):
    if dest_path.exists():
        print("  Already downloaded: {}".format(dest_path.name))
        return
    print("Downloading {} (~{} MB) ...".format(dest_path.name, size_mb))
    print("  from:", url)
    t0 = time.time()
    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print("\n  Done in {:.1f} s".format(time.time() - t0))


def extract_archive(tar_path: Path, dest_dir: Path):
    marker = dest_dir / (tar_path.stem.split('.')[0] + "_extracted.marker")
    if marker.exists():
        print("  Already extracted:", tar_path.name)
        return
    print("Extracting {} ...".format(tar_path.name))
    t0 = time.time()
    with tarfile.open(tar_path, 'r:gz') as tf:
        try:
            tf.extractall(dest_dir, filter='data')
        except TypeError:
            tf.extractall(dest_dir)
    marker.touch()
    print("  Done in {:.1f} s".format(time.time() - t0))


# LibriSpeech helpers

def classify_libri(transcript: str) -> int:
    t = transcript.lower()
    if re.search(r'\bhelp\s+me\b', t):
        return 1
    if re.search(r'\bhelp\b', t):
        return 0
    if re.search(r'\bstop\b', t):
        return 2
    return 3


def scan_librispeech(libri_root: Path, keyword_max_duration: float = None):
    """
    Scan a LibriSpeech split directory.

    keyword_max_duration: if set, keep help/help_me utterances only when
        duration_s <= this value (reduces label noise for short-window crops).
    """
    import soundfile as sf

    items = []
    trans_files = list(libri_root.rglob("*.trans.txt"))
    print("  Found {} .trans.txt files".format(len(trans_files)))

    for trans_path in trans_files:
        audio_dir = trans_path.parent
        with open(trans_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, transcript = parts
                flac_path = audio_dir / (utt_id + '.flac')
                if not flac_path.exists():
                    continue
                try:
                    info = sf.info(str(flac_path))
                    duration = info.frames / info.samplerate
                except Exception:
                    continue

                label = classify_libri(transcript)

                # Drop long keyword utterances to reduce label noise
                if keyword_max_duration and label in (0, 1):
                    if duration > keyword_max_duration:
                        continue

                items.append({
                    'speech_path': str(flac_path),
                    'kws_label':   label,
                    'duration_s':  round(duration, 3),
                    'transcript':  transcript.lower(),
                })

    return items


# Google Speech Commands helpers

def scan_speech_commands(sc_root: Path, split: str = 'train'):
    """
    Scan Google Speech Commands v2.

    split: 'train' or 'val' (uses validation_list.txt for the split boundary).

    Returns list of dicts: speech_path, kws_label, duration_s, transcript
    """
    import soundfile as sf

    val_paths  = set()
    test_paths = set()
    for fname, target in [('validation_list.txt', val_paths),
                           ('testing_list.txt',   test_paths)]:
        p = sc_root / fname
        if p.exists():
            target.update(p.read_text(encoding='utf-8').strip().splitlines())

    all_words = list(SC_KEYWORD_MAP.keys()) + SC_BACKGROUND_WORDS
    items = []

    for word in all_words:
        word_dir = sc_root / word
        if not word_dir.exists():
            continue
        label = SC_KEYWORD_MAP.get(word, 3)

        for wav_path in sorted(word_dir.glob('*.wav')):
            rel = '{}/{}'.format(word, wav_path.name)
            if split == 'train':
                if rel in val_paths or rel in test_paths:
                    continue
            else:  # val
                if rel not in val_paths:
                    continue

            try:
                info = sf.info(str(wav_path))
                duration = info.frames / info.samplerate
            except Exception:
                continue

            items.append({
                'speech_path': str(wav_path),
                'kws_label':   label,
                'duration_s':  round(duration, 3),
                'transcript':  word,
            })

    return items


# Manifest creation

def build_hybrid_manifests(libri_train, libri_val, sc_train, sc_val):
    """
    Combine LibriSpeech and Speech Commands into balanced train/val manifests.

    Speech Commands provides high-quality isolated clips for stop (label 2)
    and background hard-negatives (label 3).
    LibriSpeech provides help/help_me (labels 0/1) and additional background.
    """
    rng = np.random.default_rng(42)

    # Separate by class
    def by_class(items):
        d = {i: [] for i in range(4)}
        for it in items:
            d[it['kws_label']].append(it)
        return d

    tr = by_class(libri_train)
    sc = by_class(sc_train)
    vl = by_class(libri_val)
    sv = by_class(sc_val)

    # Train assembly
    # Stop: from Speech Commands only (perfect quality)
    stop_train = sc[2]
    rng.shuffle(stop_train)
    n_stop = len(stop_train)

    # Background: SC non-stop words + some LibriSpeech bg
    bg_sc    = sc[3]
    bg_libri = tr[3]
    rng.shuffle(bg_sc)
    rng.shuffle(bg_libri)
    bg_train = bg_sc[:n_stop * BG_SC_MULTIPLIER] + bg_libri[:BG_LIBRI_EXTRA]

    # Help / help_me: LibriSpeech short utterances only
    help_tr    = tr[0]
    help_me_tr = tr[1]
    rng.shuffle(help_tr)
    rng.shuffle(help_me_tr)

    train_items = help_tr + help_me_tr + stop_train + bg_train
    rng.shuffle(train_items)

    # Val assembly
    # All LibriSpeech dev-clean + Speech Commands val stop + SC val bg
    sc_stop_val = sv[2]
    sc_bg_val   = sv[3][:len(sc_stop_val) * 2]  # 2× stop count
    val_items   = libri_val + sc_stop_val + sc_bg_val

    # Print distribution
    print("\n  Train distribution:")
    tr_counts = Counter(i['kws_label'] for i in train_items)
    for lbl, name in enumerate(KWS_CLASSES):
        print("    {:12s}: {}".format(name, tr_counts.get(lbl, 0)))

    print("  Val distribution:")
    va_counts = Counter(i['kws_label'] for i in val_items)
    for lbl, name in enumerate(KWS_CLASSES):
        print("    {:12s}: {}".format(name, va_counts.get(lbl, 0)))

    return train_items, val_items


def write_manifest(items, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,
            fieldnames=['speech_path', 'kws_label', 'duration_s', 'transcript'])
        writer.writeheader()
        writer.writerows(items)
    print("  Written {} rows -> {}".format(len(items), path))


# Status check

def check_status(libri_dir, sc_dir, manifest_dir):
    print("=== Status ===")
    for name, (filename, _) in LIBRI_ARCHIVES.items():
        p = libri_dir / filename
        status = "OK ({:.0f} MB)".format(p.stat().st_size / 1e6) if p.exists() else "NOT FOUND"
        print("  {:<26} {}".format(filename, status))
    sc_tar = libri_dir / "speech_commands_v0.02.tar.gz"
    status = "OK ({:.0f} MB)".format(sc_tar.stat().st_size / 1e6) if sc_tar.exists() else "NOT FOUND"
    print("  {:<26} {}".format("speech_commands_v0.02.tar.gz", status))
    for split in ['train', 'val']:
        p = manifest_dir / (split + '.csv')
        if p.exists():
            import pandas as pd
            df = pd.read_csv(p)
            c  = Counter(df['kws_label'])
            print("  manifests/{}.csv  {} rows  {}".format(
                split, len(df),
                {KWS_CLASSES[i]: c.get(i, 0) for i in range(4)}))
        else:
            print("  manifests/{}.csv    NOT FOUND".format(split))


# Main

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dev-only', action='store_true',
                        help='LibriSpeech dev-clean only, no Speech Commands (quick test)')
    parser.add_argument('--check', action='store_true',
                        help='Show what is already downloaded, do not download')
    args = parser.parse_args()

    script_dir   = Path(__file__).parent
    data_dir     = script_dir / 'data'
    libri_dir    = data_dir / 'librispeech'
    sc_dir       = data_dir / SC_DIRNAME
    manifest_dir = data_dir / 'manifests'
    libri_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_status(libri_dir, sc_dir, manifest_dir)
        return

    # Download LibriSpeech
    print("=" * 60)
    print("Step 1/3 — LibriSpeech")
    print("=" * 60)
    archives = ['dev-clean'] if args.dev_only else ['train-clean-100', 'dev-clean']
    for name in archives:
        filename, size = LIBRI_ARCHIVES[name]
        tar_path = libri_dir / filename
        download_file(LIBRI_BASE_URL + filename, tar_path, size)
        extract_archive(tar_path, libri_dir)

    # Download Speech Commands
    use_sc = not args.dev_only
    if use_sc:
        print("\n" + "=" * 60)
        print("Step 2/3 — Google Speech Commands v2")
        print("=" * 60)
        sc_tar = libri_dir / "speech_commands_v0.02.tar.gz"
        download_file(SC_URL, sc_tar, SC_SIZE_MB)
        # Archive extracts flat — word dirs land directly in data_dir
        extract_archive(sc_tar, data_dir)

    # Scan LibriSpeech
    print("\n" + "=" * 60)
    print("Step 3/3 — Scanning & building manifests")
    print("=" * 60)

    libri_root = libri_dir / 'LibriSpeech'

    print("\n[LibriSpeech train]")
    if not args.dev_only and (libri_root / 'train-clean-100').exists():
        libri_train_items = scan_librispeech(
            libri_root / 'train-clean-100',
            keyword_max_duration=MAX_KEYWORD_DURATION_S)
    else:
        print("  train-clean-100 not found, using dev-clean split 80/20")
        all_items = scan_librispeech(libri_root / 'dev-clean',
                                     keyword_max_duration=MAX_KEYWORD_DURATION_S)
        rng = np.random.default_rng(42)
        rng.shuffle(all_items)
        n = int(len(all_items) * 0.8)
        libri_train_items, libri_val_items = all_items[:n], all_items[n:]

    print("  Total: {}".format(len(libri_train_items)))

    print("\n[LibriSpeech val (dev-clean)]")
    if (libri_root / 'dev-clean').exists():
        libri_val_items = scan_librispeech(libri_root / 'dev-clean')
    else:
        libri_val_items = []
    print("  Total: {}".format(len(libri_val_items)))

    # Scan Speech Commands
    sc_train_items = []
    sc_val_items   = []

    if use_sc:
        # Detect where Speech Commands landed: preferred subdir or flat in data_dir
        if (data_dir / SC_DIRNAME / 'stop').exists():
            sc_dir = data_dir / SC_DIRNAME
        elif (data_dir / 'stop').exists():
            sc_dir = data_dir   # flat extraction (common for this archive)
        else:
            sc_dir = None
            print("\n  WARNING: Speech Commands 'stop' directory not found after extraction.")

        if sc_dir is not None:
            print("\n[Speech Commands train]  root={}".format(sc_dir))
            sc_train_items = scan_speech_commands(sc_dir, split='train')
            print("  Total: {}".format(len(sc_train_items)))

            print("\n[Speech Commands val]")
            sc_val_items = scan_speech_commands(sc_dir, split='val')
            print("  Total: {}".format(len(sc_val_items)))

    # Build manifests
    print("\n[Building manifests]")

    if use_sc and sc_train_items and sc_val_items:
        train_items, val_items = build_hybrid_manifests(
            libri_train_items, libri_val_items,
            sc_train_items,    sc_val_items)
    else:
        # dev-only or no SC: fall back to LibriSpeech-only with simple balance
        rng2 = np.random.default_rng(42)
        by_cls = {i: [] for i in range(4)}
        for it in libri_train_items:
            by_cls[it['kws_label']].append(it)
        kw_counts  = [len(by_cls[i]) for i in range(3)]
        nonzero    = [c for c in kw_counts if c > 0]
        bg_cap     = (min(nonzero) * 6) if nonzero else len(by_cls[3])
        rng2.shuffle(by_cls[3])
        by_cls[3] = by_cls[3][:bg_cap]
        train_items = sum(by_cls.values(), [])
        rng2.shuffle(train_items)
        val_items   = libri_val_items

        print("\n  Train distribution (LibriSpeech only):")
        c = Counter(i['kws_label'] for i in train_items)
        for lbl, name in enumerate(KWS_CLASSES):
            print("    {:12s}: {}".format(name, c.get(lbl, 0)))

    write_manifest(train_items, manifest_dir / 'train.csv')
    write_manifest(val_items,   manifest_dir / 'val.csv')

    print("\nDone.  Next: python run_training.py")


if __name__ == '__main__':
    main()
