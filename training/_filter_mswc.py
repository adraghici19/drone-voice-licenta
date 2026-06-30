"""Filter MSWC 'help' clips with Whisper ASR: keep only clips whose transcription
contains 'help'. Produces a clean, verified multi-speaker set per split.
Preserves the speaker-disjoint train/dev/test structure."""
import os, glob, shutil, time, csv
import numpy as np
import soundfile as sf
import librosa
import whisper

SRC = os.path.join("data", "mswc_help")
DST = os.path.join("data", "mswc_help_clean")
model = whisper.load_model("base.en")
print("loaded base.en", flush=True)

report = open(os.path.join("data", "mswc_filter_report.csv"), "w", newline="", encoding="utf-8")
w = csv.writer(report); w.writerow(["split", "file", "transcript", "kept"])

for split in ["test", "dev", "train"]:
    files = sorted(glob.glob(os.path.join(SRC, split, "*.wav")))
    out = os.path.join(DST, split); os.makedirs(out, exist_ok=True)
    kept = 0; t0 = time.time()
    print("=== {} : {} clips ===".format(split, len(files)), flush=True)
    for i, f in enumerate(files):
        try:
            x, sr = sf.read(f, dtype="float32")
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            txt = model.transcribe(x.astype(np.float32), language="en", fp16=False)["text"].strip().lower()
        except Exception as e:
            txt = "ERR:" + str(e)[:30]
        keep = "help" in txt
        w.writerow([split, os.path.basename(f), txt, int(keep)])
        if keep:
            shutil.copy(f, os.path.join(out, os.path.basename(f)))
            kept += 1
        if i % 200 == 0 and i > 0:
            el = time.time() - t0
            print("  {}/{} kept {} ({:.0f}s, ETA {:.0f}s)".format(
                i, len(files), kept, el, el / i * (len(files) - i)), flush=True)
    report.flush()
    print("  {} : kept {}/{} ({:.0f}%) in {:.0f}s".format(
        split, kept, len(files), 100 * kept / max(len(files), 1), time.time() - t0), flush=True)
report.close()
print("DONE -> {}".format(DST), flush=True)
