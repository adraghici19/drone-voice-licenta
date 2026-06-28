import os, glob, csv, random
import soundfile as sf

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MAN = os.path.join(DATA, "manifests")
os.makedirs(MAN, exist_ok=True)

ESC_EXCLUDE = {"breathing", "coughing", "sneezing", "laughing", "crying_baby", "snoring"}
rows = {"train": [], "val": [], "test": []}


def dur(path):
    try:
        info = sf.info(path); return round(info.frames / info.samplerate, 3)
    except Exception:
        return 1.0


def add(split, paths, label, is_voice, cap=None):
    random.shuffle(paths)
    if cap:
        paths = paths[:cap]
    for p in paths:
        rows[split].append((os.path.abspath(p), label, dur(p), is_voice))
    return len(paths)


def split_files(files, frac_val=0.1, frac_test=0.1, seed=0):
    f = list(files); random.Random(seed).shuffle(f)
    n = len(f); ntr = int(n * (1 - frac_val - frac_test)); nva = int(n * frac_val)
    return f[:ntr], f[ntr:ntr + nva], f[ntr + nva:]


mswc = lambda s: glob.glob(os.path.join(DATA, "mswc_help_clean", s, "*.wav"))
user_help = glob.glob(os.path.join(DATA, "tts_keywords", "help", "rec_*.wav"))
print("help  train:", add("train", mswc("train") + user_help, 0, 1, 2000))
print("help  val  :", add("val", mswc("dev"), 0, 1, 300))
print("help  test :", add("test", mswc("test"), 0, 1, 300))

val_set = set(open(os.path.join(DATA, "validation_list.txt")).read().split())
test_set = set(open(os.path.join(DATA, "testing_list.txt")).read().split())
s_tr, s_va, s_te = [], [], []
for p in glob.glob(os.path.join(DATA, "stop", "*.wav")):
    rel = "stop/" + os.path.basename(p)
    (s_te if rel in test_set else s_va if rel in val_set else s_tr).append(p)
print("stop  train:", add("train", s_tr, 1, 1, 2000))
print("stop  val  :", add("val", s_va, 1, 1, 300))
print("stop  test :", add("test", s_te, 1, 1, 300))

LS = os.path.join(DATA, "librispeech", "LibriSpeech")
train100 = glob.glob(os.path.join(LS, "train-clean-100", "*", "*", "*.flac"))
devclean = glob.glob(os.path.join(LS, "dev-clean", "*", "*", "*.flac"))
spk = lambda f: os.path.normpath(f).split(os.sep)[-3]
speakers = sorted({spk(f) for f in train100}); random.shuffle(speakers)
val_spk = set(speakers[: max(1, len(speakers) // 10)])
print("speech train (LibriSpeech):", add("train", [f for f in train100 if spk(f) not in val_spk], 2, 1, 1700))
print("speech val   (LibriSpeech):", add("val", [f for f in train100 if spk(f) in val_spk], 2, 1, 200))
print("speech test  (LibriSpeech):", add("test", devclean, 2, 1, 200))
m_sp = glob.glob(os.path.join(DATA, "musan", "speech", "**", "*.wav"), recursive=True)
tr, va, te = split_files(m_sp, seed=1)
print("speech train (MUSAN):", add("train", tr, 2, 1, 900))
print("speech val   (MUSAN):", add("val", va, 2, 1, 120))
print("speech test  (MUSAN):", add("test", te, 2, 1, 120))

fold = {"train": [], "val": [], "test": []}
with open(os.path.join(DATA, "esc50", "esc50.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["category"] in ESC_EXCLUDE:
            continue
        p = os.path.join(DATA, "esc50", "audio", r["filename"]); fo = int(r["fold"])
        fold["train" if fo in (1, 2, 3) else "val" if fo == 4 else "test"].append(p)
print("noise train (ESC-50):", add("train", fold["train"], 3, 0))
print("noise val   (ESC-50):", add("val", fold["val"], 3, 0))
print("noise test  (ESC-50):", add("test", fold["test"], 3, 0))
for sub, cap, sd in [("noise", 800, 2), ("music", 450, 3)]:
    files = glob.glob(os.path.join(DATA, "musan", sub, "**", "*.wav"), recursive=True)
    tr, va, te = split_files(files, seed=sd)
    print("noise train (MUSAN {}):".format(sub), add("train", tr, 3, 0, cap))
    print("noise val   (MUSAN {}):".format(sub), add("val", va, 3, 0, 120))
    print("noise test  (MUSAN {}):".format(sub), add("test", te, 3, 0, 120))

names = ["help", "stop", "speech", "noise"]
for split in ["train", "val", "test"]:
    random.shuffle(rows[split])
    with open(os.path.join(MAN, split + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speech_path", "kws_label", "duration_s", "is_voice"])
        w.writerows(rows[split])
    from collections import Counter
    c = Counter(r[1] for r in rows[split])
    print("{:5s}: total {}  ".format(split, len(rows[split])) +
          "  ".join("{}={}".format(names[i], c.get(i, 0)) for i in range(4)))
print("\nDONE -> manifests train/val/test.csv")
