"""Download + extract ESC-50 (environmental sound dataset) for the 'noise' class."""
import os, urllib.request, zipfile, io, csv, collections

URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
DST = os.path.join("data", "esc50")
os.makedirs(DST, exist_ok=True)

print("Downloading ESC-50 (~600 MB) ...", flush=True)
data = urllib.request.urlopen(URL, timeout=300).read()
print("downloaded {:.0f} MB, extracting ...".format(len(data) / 1e6), flush=True)

n_audio = 0
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for m in z.namelist():
        if m.endswith(".wav") and "/audio/" in m:
            out = os.path.join(DST, "audio", os.path.basename(m))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as src, open(out, "wb") as dst:
                dst.write(src.read())
            n_audio += 1
        elif m.endswith("meta/esc50.csv"):
            with z.open(m) as src:
                open(os.path.join(DST, "esc50.csv"), "wb").write(src.read())

print("extracted {} wav files".format(n_audio), flush=True)
# list categories
cats = collections.Counter()
with open(os.path.join(DST, "esc50.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cats[row["category"]] += 1
print("categories ({}):".format(len(cats)))
for c in sorted(cats):
    print("  {:24s} {}".format(c, cats[c]))
