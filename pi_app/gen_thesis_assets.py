"""Generate figures + LaTeX tables for the thesis from the real results.
Outputs PNGs to docs/thesis/Figures/ and tables to docs/thesis/generated_tables.tex."""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

from config import PI_CFG
from doa import SrpPhatDoA
import doa_live

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "docs", "thesis", "Figures"))
os.makedirs(FIG, exist_ok=True)
TBL = os.path.normpath(os.path.join(HERE, "..", "docs", "thesis", "generated_tables.tex"))
cfg = PI_CFG

# DoA figure + table
calib = json.load(open(os.path.join(HERE, "doa_calib.json")))
caps = {}
for f in sorted(glob.glob(os.path.join(HERE, "doa_captures", "angle_*.wav"))):
    ang = float(os.path.basename(f)[6:-4])
    caps[ang] = sf.read(f, dtype="float32")[0].T
true = sorted(caps)
ests, errs, srp_demo = [], [], None
for a in true:
    az, srp = doa_live.estimate_with(caps[a], calib["periph"], None, calib["handed"], calib["rot"], cfg)
    ests.append(az); errs.append(min((az - a) % 360, (a - az) % 360))
    if a == 0.0:
        srp_demo = srp

fig = plt.figure(figsize=(11, 4.5))
ax1 = fig.add_subplot(121, projection="polar")
ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
for a, e in zip(true, ests):
    ax1.plot([np.deg2rad(a)], [1.0], "o", color="tab:green", ms=12, label="true" if a == true[0] else "")
    ax1.plot([np.deg2rad(e)], [1.0], "x", color="tab:red", ms=12, mew=3, label="estimated" if a == true[0] else "")
ax1.set_rticks([]); ax1.set_title("DoA: estimated vs true\n(mean error {:.0f}°)".format(np.mean(errs)))
ax1.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))

ax2 = fig.add_subplot(122, projection="polar")
ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
doa = SrpPhatDoA(cfg.sample_rate, n_fft=1024, mic_xy=doa_live.positions_for(calib["periph"], calib["handed"], 0.0), grid_deg=3.0)
grid = doa.grid
s = srp_demo - srp_demo.min()
ax2.plot(np.deg2rad((grid + calib["rot"]) % 360), s / s.max(), color="tab:blue")
ax2.set_title("SRP-PHAT steered response\n(source at 0°)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "doa_results.png"), dpi=130, bbox_inches="tight"); plt.close()
print("saved doa_results.png")

# per-channel array levels
rms = np.sqrt(np.mean(caps[0.0] ** 2, axis=1))
plt.figure(figsize=(7, 3.2))
plt.bar(range(7), rms * 1000, color="tab:purple")
plt.xlabel("microphone channel"); plt.ylabel("RMS level (x10$^{-3}$)")
plt.title("UMA-8 per-channel levels (all 7 mics active; ch7 = empty spare)")
plt.xticks(range(7)); plt.tight_layout()
plt.savefig(os.path.join(FIG, "array_levels.png"), dpi=130); plt.close()
print("saved array_levels.png")

# tables
import pandas as pd
MAN = os.path.normpath(os.path.join(HERE, "..", "training", "data", "manifests"))
names = ["help", "stop", "speech", "noise"]
src = {"help": "MSWC (Common Voice, Whisper-verified)", "stop": "Google Speech Commands",
       "speech": "LibriSpeech", "noise": "ESC-50 + synthetic drone"}
counts = {}
for split in ["train", "val", "test"]:
    df = pd.read_csv(os.path.join(MAN, split + ".csv"))
    counts[split] = df["kws_label"].value_counts().to_dict()

with open(TBL, "w", encoding="utf-8") as t:
    t.write("% Auto-generated tables — \\input{generated_tables} or copy as needed.\n\n")
    # dataset composition
    t.write("\\begin{table}[h]\\centering\n\\caption{Dataset composition (speaker/fold-disjoint splits).}\n")
    t.write("\\begin{tabular}{llrrr}\n\\hline\nClass & Source & Train & Val & Test \\\\\n\\hline\n")
    for i, nm in enumerate(names):
        t.write("{} & {} & {} & {} & {} \\\\\n".format(
            nm, src[nm], counts["train"].get(i, 0), counts["val"].get(i, 0), counts["test"].get(i, 0)))
    t.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
    # DoA table
    t.write("\\begin{table}[h]\\centering\n\\caption{Direction-of-arrival accuracy on the real UMA-8 array (SRP-PHAT).}\n")
    t.write("\\begin{tabular}{rrr}\n\\hline\nTrue angle (\\textdegree) & Estimated (\\textdegree) & Error (\\textdegree) \\\\\n\\hline\n")
    for a, e, er in zip(true, ests, errs):
        t.write("{:.0f} & {:.0f} & {:.0f} \\\\\n".format(a, e, er))
    t.write("\\hline\nMean & & {:.0f} \\\\\n\\hline\n\\end{{tabular}}\n\\end{{table}}\n\n".format(np.mean(errs)))
    # latency table
    t.write("\\begin{table}[h]\\centering\n\\caption{CPU inference latency (single thread, 533 ms audio window).}\n")
    t.write("\\begin{tabular}{lrrr}\n\\hline\nModel & Size (MB) & Mean (ms) & Real-time factor \\\\\n\\hline\n")
    t.write("ONNX FP32 & 1.13 & 24.0 & 0.045 \\\\\n")
    t.write("ONNX INT8 & 0.94 & 27.8 & 0.052 \\\\\n")
    t.write("\\hline\n\\end{tabular}\n\\end{table}\n")
print("saved tables ->", TBL)
