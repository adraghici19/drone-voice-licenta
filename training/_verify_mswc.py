"""Objectively verify the MSWC 'help' clips actually say 'help', using Whisper ASR
on a random sample. Also copy a few clips out so they can be listened to by ear."""
import os, glob, random, shutil
import numpy as np
import soundfile as sf
import librosa
import whisper

random.seed(0)
model = whisper.load_model("tiny.en")

listen_dir = os.path.join("data", "mswc_help", "_listen_samples")
os.makedirs(listen_dir, exist_ok=True)

for split in ["train", "test"]:
    files = glob.glob(os.path.join("data", "mswc_help", split, "*.wav"))
    sample = random.sample(files, min(20, len(files)))
    hits = 0
    print("=== {} : transcribing {} random clips ===".format(split, len(sample)))
    for i, f in enumerate(sample):
        x, sr = sf.read(f, dtype="float32")
        if sr != 16000:
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        r = model.transcribe(x.astype(np.float32), language="en", fp16=False)
        txt = r["text"].strip().lower()
        is_help = "help" in txt
        hits += is_help
        if i < 12:
            print("  '{}'  {}".format(txt, "OK" if is_help else "<-- NOT help?"))
        # copy first 8 of train to a listen folder
        if split == "train" and i < 8:
            shutil.copy(f, os.path.join(listen_dir, os.path.basename(f)))
    print("  -> {}/{} transcribed containing 'help' ({:.0f}%)".format(hits, len(sample), 100 * hits / len(sample)))
print("\n8 sample clips copied to {} for you to listen.".format(listen_dir))
