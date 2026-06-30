"""
Fine-tune LiteVoiceNet on the newly-recorded 'help' clips.

Loads the existing best checkpoint and trains for a small number of epochs
with a reduced learning rate so the model learns the new voice without
forgetting 'stop' / noise discrimination.

Run:  python finetune.py
Then: python export.py          (re-exports ONNX + INT8)
"""

import sys, os, warnings, time
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib; matplotlib.use("Agg")
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import CFG
from model import build_model
from data import build_datasets, ManifestDataset
from losses import build_loss
import export as exp_mod

# Fine-tune settings
FINETUNE_EPOCHS = 12
LR              = 1.5e-4   # ~4x lower than original 6e-4
BATCH           = 64
CKPT_IN         = os.path.join(CFG.export_dir, "litevoicenet_best.pt")
CKPT_OUT        = os.path.join(CFG.export_dir, "litevoicenet_best.pt")

# Setup
print("=" * 60)
print("LiteVoiceNet  —  fine-tune on new 'help' recordings")
print("=" * 60)

if not torch.cuda.is_available():
    print("[warn] CUDA not found — running on CPU (slower)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
np.random.seed(42)

# Data
train_ds, val_ds = build_datasets(CFG)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
print(f"Train  : {len(train_ds)} samples  ({len(train_loader)} batches)")
print(f"Val    : {len(val_ds)} samples  ({len(val_loader)} batches)")

# Model
model = build_model(CFG).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params : {n_params:,}")

if os.path.exists(CKPT_IN):
    ckpt  = torch.load(CKPT_IN, map_location=device, weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    epoch_info = f"  (saved at epoch {ckpt['epoch']})" if isinstance(ckpt, dict) and "epoch" in ckpt else ""
    print(f"Loaded : {CKPT_IN}{epoch_info}")
else:
    print(f"[warn] No checkpoint at {CKPT_IN} — training from scratch")

# Loss + Optimizer
from collections import Counter
counts  = Counter(train_ds.df["kws_label"].astype(int).tolist())
n_total = len(train_ds)
n_cls   = CFG.model.num_kws_classes
kws_weight = torch.tensor([
    (n_total / (n_cls * max(counts.get(i, 1), 1))) ** 0.5
    for i in range(n_cls)
], dtype=torch.float32)
kws_weight = kws_weight / kws_weight.sum() * n_cls
print("KWS class weights:", ["{}={:.2f}".format(CFG.model.kws_classes[i], kws_weight[i].item())
                              for i in range(n_cls)])

criterion = build_loss(CFG, kws_weight=kws_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=LR/10)

# Training loop
def run_epoch(loader, train=True):
    model.train(train)
    tot_loss, kws_correct, kws_total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for feat, mask_t, vad_t, kws_t in loader:
            feat   = feat.to(device, non_blocking=True)
            mask_t = mask_t.to(device, non_blocking=True)
            vad_t  = vad_t.to(device, non_blocking=True)
            kws_t  = kws_t.to(device, non_blocking=True)

            preds = model(feat)
            loss, _ = criterion(preds, (mask_t, vad_t, kws_t))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tot_loss += loss.item() * len(feat)
            kws_p = preds[2]
            kws_correct += (kws_p.argmax(dim=-1) == kws_t).sum().item()
            kws_total   += len(kws_t)

    return tot_loss / len(loader.dataset), kws_correct / max(kws_total, 1)


best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "train_kws": [], "val_kws": []}

print("\nEpoch  Train-Loss  Val-Loss  Train-KWS  Val-KWS")
print("-" * 52)

for ep in range(1, FINETUNE_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_kws = run_epoch(train_loader, train=True)
    va_loss, va_kws = run_epoch(val_loader,   train=False)
    scheduler.step()
    elapsed = time.time() - t0

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_kws"].append(tr_kws)
    history["val_kws"].append(va_kws)

    marker = " *" if va_kws > best_val_acc else ""
    print(f"  {ep:2d}    {tr_loss:.4f}      {va_loss:.4f}    "
          f"{tr_kws*100:.1f}%      {va_kws*100:.1f}%   ({elapsed:.0f}s){marker}")

    if va_kws > best_val_acc:
        best_val_acc = va_kws
        torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_total": va_loss}, CKPT_OUT)

print(f"\nBest val KWS: {best_val_acc*100:.1f}%  ->  {CKPT_OUT}")

# Loss curves
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("epoch")
axes[1].plot([x*100 for x in history["train_kws"]], label="train")
axes[1].plot([x*100 for x in history["val_kws"]],   label="val")
axes[1].set_title("KWS accuracy (%)"); axes[1].legend(); axes[1].set_xlabel("epoch")
fig.tight_layout()
out_fig = os.path.join(CFG.export_dir, "finetune_curves.png")
fig.savefig(out_fig, dpi=100)
print(f"Curves : {out_fig}")

# Re-export ONNX
print("\nExporting ONNX ...")
ckpt = torch.load(CKPT_OUT, map_location=device, weights_only=True)
model.load_state_dict(ckpt["state_dict"])
model_cpu = model.cpu()
onnx_path = exp_mod.export_onnx(model_cpu, CFG)
exp_mod.quantize_onnx_int8(onnx_path, cfg=CFG)
print("Done. Run  python ../pi_app/combined.py --device 25  to test.")
