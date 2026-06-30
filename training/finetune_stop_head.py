import sys, os, copy, shutil, datetime
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

from config import CFG
from model import build_model
from data import build_datasets
import export as exp_mod

EPOCHS   = 15
LR       = 5e-4
BATCH    = 64
CKPT_IN  = os.path.join(CFG.export_dir, 'litevoicenet_best.pt')
CKPT_OUT = os.path.join(CFG.export_dir, 'litevoicenet_best.pt')
STOP_WEIGHT_MULT = 5.0

print("=" * 60)
print("LiteVoiceNet - KWS-head fine-tune for STOP")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

train_ds, val_ds = build_datasets(CFG)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)
print("Train: {}  Val: {}".format(len(train_ds), len(val_ds)))

model = build_model(CFG).to(device)
ckpt  = torch.load(CKPT_IN, map_location='cpu')
state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
model.load_state_dict(state)
print("Loaded checkpoint from epoch {} (val={:.4f})".format(
    ckpt.get('epoch', '?'), ckpt.get('val_total', float('nan'))))

archive_dir = os.path.join(CFG.export_dir, 'archive')
os.makedirs(archive_dir, exist_ok=True)
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for fname in ['litevoicenet_best.pt', 'litevoicenet.onnx']:
    src = os.path.join(CFG.export_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(archive_dir, '{}_{}'.format(ts, fname)))
print("Archived to {}/{}*".format(archive_dir, ts))

for name, param in model.named_parameters():
    param.requires_grad = 'kws_head' in name

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print("Trainable params: {:,} / {:,}".format(trainable, total))

counts = Counter(train_ds.df['kws_label'].astype(int).tolist())
n_total = len(train_ds)
n_cls   = CFG.model.num_kws_classes
kws_weight = torch.tensor([
    (n_total / (n_cls * max(counts.get(i, 1), 1))) ** 0.5
    for i in range(n_cls)
], dtype=torch.float32)
kws_weight = kws_weight / kws_weight.sum() * n_cls
kws_weight[1] *= STOP_WEIGHT_MULT
kws_weight = kws_weight / kws_weight.sum() * n_cls
print("KWS weights: {}".format(
    ["{}={:.2f}".format(CFG.model.kws_classes[i], kws_weight[i].item()) for i in range(n_cls)]))
kws_weight = kws_weight.to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val   = float('inf')
best_state = None
best_epoch = 0

print("\n{:<6} {:<12} {:<12}  {:<8} {:<8} {:<8} {:<8}".format(
    "Epoch", "Train-KWS", "Val-KWS", "help%", "stop%", "speech%", "noise%"))
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_kws = 0.0; n_tr = 0
    for feat, mask_t, vad_t, kws_t in train_loader:
        feat  = feat.to(device, non_blocking=True)
        kws_t = kws_t.to(device, non_blocking=True)
        optimizer.zero_grad()
        preds = model(feat)
        loss = F.nll_loss(preds[2], kws_t, weight=kws_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 1.0)
        optimizer.step()
        t_kws += loss.item(); n_tr += 1
    scheduler.step()

    model.eval()
    v_kws = 0.0; n_val = 0
    pred_all, true_all = [], []
    with torch.no_grad():
        for feat, mask_t, vad_t, kws_t in val_loader:
            feat  = feat.to(device, non_blocking=True)
            kws_t = kws_t.to(device, non_blocking=True)
            preds = model(feat)
            loss = F.nll_loss(preds[2], kws_t, weight=kws_weight)
            v_kws += loss.item(); n_val += 1
            pred_all.extend(preds[2].argmax(dim=1).cpu().tolist())
            true_all.extend(kws_t.cpu().tolist())
    torch.cuda.empty_cache()

    cc = Counter(); tc = Counter()
    for p, t in zip(pred_all, true_all):
        tc[t] += 1
        if p == t: cc[t] += 1
    per_cls = [100 * cc.get(i, 0) / max(tc.get(i, 1), 1) for i in range(n_cls)]

    v_loss = v_kws / n_val
    marker = ""
    if v_loss < best_val:
        best_val = v_loss; best_epoch = epoch
        best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        torch.save({'state_dict': best_state, 'epoch': best_epoch, 'val_total': best_val}, CKPT_OUT)
        marker = " *"

    print("{:<6} {:<12.4f} {:<12.4f}  {:>6.1f}% {:>6.1f}% {:>7.1f}% {:>6.1f}%{}".format(
        epoch, t_kws/n_tr, v_loss,
        per_cls[0], per_cls[1], per_cls[2], per_cls[3], marker))

print("\nBest epoch: {}  val_kws_loss={:.4f}".format(best_epoch, best_val))

model.load_state_dict(best_state)
model.eval()

print("\nExporting ONNX...")
model_cpu = model.cpu().eval()
onnx_path = os.path.join(CFG.export_dir, 'litevoicenet.onnx')
exp_mod.export_onnx(model_cpu, CFG, output_path=onnx_path, seq_frames=CFG.synth.seq_frames)
print("ONNX export OK ->", onnx_path)
