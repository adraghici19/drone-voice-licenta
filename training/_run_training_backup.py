"""Full training script — same logic as the notebook, console-friendly."""
import sys, os
# Reduce CUDA fragmentation so large batches (224) fit on the 24 GB RTX 3090.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves PNGs instead of showing windows

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import CFG
from model import build_model
from data import build_datasets
from losses import build_loss
import export as exp_mod

# ── Env check ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("LiteVoiceNet — full training run")
print("=" * 60)
print("PyTorch :", torch.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — check your PyTorch installation.")
gpu = torch.cuda.get_device_properties(0)
print("GPU     :", gpu.name)
print("VRAM    : {:.1f} GB".format(gpu.total_memory / 1e9))
device = torch.device('cuda')

# ── Dataset ───────────────────────────────────────────────────────────────────
torch.manual_seed(CFG.train.seed)
np.random.seed(CFG.train.seed)
train_ds, val_ds = build_datasets(CFG)
train_loader = DataLoader(train_ds, batch_size=CFG.train.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG.train.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

# Held-out TEST set (speaker-disjoint) — used only for the final honest report,
# never for checkpoint selection.
from data import ManifestDataset
test_csv = os.path.join(CFG.data_dir, 'manifests', 'test.csv')
test_loader = None
if os.path.exists(test_csv):
    test_ds = ManifestDataset(test_csv, CFG, augment=False)
    test_loader = DataLoader(test_ds, batch_size=CFG.train.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    print("Test  : {} samples  (held-out, speaker-disjoint)".format(len(test_ds)))
print("\nTrain : {} samples  ({} batches)".format(len(train_ds), len(train_loader)))
print("Val   : {} samples  ({} batches)".format(len(val_ds),   len(val_loader)))

# ── Model ─────────────────────────────────────────────────────────────────────
model = build_model(CFG).to(device)
n_params = sum(p.numel() for p in model.parameters())
assert n_params < 1_500_000, "Exceeds 1.5 M budget: {:,}".format(n_params)
print("\nParams  : {:,}  ({:.1f}% of 1.5 M budget)".format(n_params, n_params / 1_500_000 * 100))

# ── KWS class weights (inverse-frequency, only when using real manifests) ────
kws_weight = None
if hasattr(train_ds, 'df'):
    from collections import Counter
    counts   = Counter(train_ds.df['kws_label'].astype(int).tolist())
    n_total  = len(train_ds)
    n_cls    = CFG.model.num_kws_classes
    # sqrt of inverse-frequency → raport max/min ≈ 10:1 (liniarul dădea 100:1)
    kws_weight = torch.tensor([
        (n_total / (n_cls * max(counts.get(i, 1), 1))) ** 0.5
        for i in range(n_cls)
    ], dtype=torch.float32)
    kws_weight = kws_weight / kws_weight.sum() * n_cls  # keep scale ≈ 1 on average
    print("KWS class weights:", ["{}={:.2f}".format(CFG.model.kws_classes[i], kws_weight[i].item())
                                  for i in range(n_cls)])

# ── Training setup ────────────────────────────────────────────────────────────
criterion = build_loss(CFG, kws_weight=kws_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.train.learning_rate,
                             weight_decay=CFG.train.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG.train.num_epochs, eta_min=1e-5)

history = {k: [] for k in ['train_total', 'train_mask', 'train_vad', 'train_kws',
                             'val_total',   'val_mask',   'val_vad',   'val_kws']}
EPOCHS   = CFG.train.num_epochs
import copy
best_val   = float('inf')      # track best val_total to avoid exporting an overfit model
best_state = None
best_epoch = 0
os.makedirs(CFG.export_dir, exist_ok=True)
ckpt_path  = os.path.join(CFG.export_dir, 'litevoicenet_best.pt')  # saved on every improvement
LOG_FREQ = max(1, EPOCHS // 10)
t_start  = time.time()

print("\n{:<8} {:<12} {:<8} {:<8} {:<8}  {:<12} {:<8} {:<8} {:<8}  {:>6}".format(
    "Epoch", "Train", "Mask", "VAD", "KWS", "Val", "Mask", "VAD", "KWS", "Time"))
print("-" * 95)

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    t_tot = t_mask = t_vad = t_kws = 0.0; n_tr = 0
    for feat, mask_tgt, vad_tgt, kws_tgt in train_loader:
        feat     = feat.to(device, non_blocking=True)
        mask_tgt = mask_tgt.to(device, non_blocking=True)
        vad_tgt  = vad_tgt.to(device, non_blocking=True)
        kws_tgt  = kws_tgt.to(device, non_blocking=True)
        optimizer.zero_grad()
        preds = model(feat)
        loss, bd = criterion(preds, (mask_tgt, vad_tgt, kws_tgt))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip)
        optimizer.step()
        t_tot += loss.item(); t_mask += bd['mask']
        t_vad += bd['vad'];   t_kws  += bd['kws']; n_tr += 1
    scheduler.step()

    model.eval()
    v_tot = v_mask = v_vad = v_kws = 0.0; n_val = 0
    with torch.no_grad():
        for feat, mask_tgt, vad_tgt, kws_tgt in val_loader:
            feat     = feat.to(device, non_blocking=True)
            mask_tgt = mask_tgt.to(device, non_blocking=True)
            vad_tgt  = vad_tgt.to(device, non_blocking=True)
            kws_tgt  = kws_tgt.to(device, non_blocking=True)
            preds    = model(feat)
            loss, bd = criterion(preds, (mask_tgt, vad_tgt, kws_tgt))
            v_tot += loss.item(); v_mask += bd['mask']
            v_vad += bd['vad'];   v_kws  += bd['kws']; n_val += 1

    # ── keep the best-generalising model (lowest val_total) ──────────────────
    if v_tot / n_val < best_val:
        best_val   = v_tot / n_val
        best_epoch = epoch
        best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        # persist to disk immediately so a mid-training crash never loses progress
        torch.save({'state_dict': best_state, 'epoch': best_epoch, 'val_total': best_val}, ckpt_path)

    for k, v in [('train_total', t_tot/n_tr), ('train_mask', t_mask/n_tr),
                  ('train_vad',  t_vad/n_tr),  ('train_kws',  t_kws/n_tr),
                  ('val_total',  v_tot/n_val),  ('val_mask',  v_mask/n_val),
                  ('val_vad',    v_vad/n_val),  ('val_kws',   v_kws/n_val)]:
        history[k].append(v)

    if epoch % LOG_FREQ == 0 or epoch == 1 or epoch == EPOCHS:
        print("{:<8} {:<12.4f} {:<8.3f} {:<8.3f} {:<8.3f}  {:<12.4f} {:<8.3f} {:<8.3f} {:<8.3f}  {:>5.0f}s".format(
            epoch, t_tot/n_tr, t_mask/n_tr, t_vad/n_tr, t_kws/n_tr,
            v_tot/n_val, v_mask/n_val, v_vad/n_val, v_kws/n_val,
            time.time() - t_start))

total_time = time.time() - t_start
print("-" * 95)
print("Training complete in {:.1f} s  ({:.1f} s/epoch)".format(total_time, total_time / EPOCHS))

# ── Restore best-val checkpoint (already persisted to disk during training) ───
if best_state is not None:
    model.load_state_dict(best_state)
    print("Restored best model from epoch {} (val_total={:.4f}); checkpoint at {}".format(
        best_epoch, best_val, ckpt_path))

model.eval()
all_kws_pred_final, all_kws_true_final = [], []
calib_features = []        # REAL feature windows for INT8 calibration
n_calib = 0
N_CALIB = 200
with torch.no_grad():
    for feat, mask_tgt, vad_tgt, kws_tgt in val_loader:
        if n_calib < N_CALIB:                 # count SAMPLES, not batches
            take = feat[: N_CALIB - n_calib].cpu().numpy()
            calib_features.append(take)
            n_calib += take.shape[0]
        feat    = feat.to(device, non_blocking=True)
        preds   = model(feat)
        all_kws_pred_final.extend(preds[2].argmax(dim=1).cpu().tolist())
        all_kws_true_final.extend(kws_tgt.tolist())
calib_features = np.concatenate(calib_features, axis=0) if calib_features else None

# ── Per-class KWS accuracy ────────────────────────────────────────────────────
if all_kws_pred_final:
    from collections import Counter
    correct_cls = Counter()
    total_cls   = Counter()
    for pred, true in zip(all_kws_pred_final, all_kws_true_final):
        total_cls[true] += 1
        if pred == true:
            correct_cls[true] += 1
    overall_acc = sum(correct_cls.values()) / max(sum(total_cls.values()), 1)
    print("\nKWS accuracy on val (best epoch {}):  overall = {:.1f}%".format(best_epoch, overall_acc * 100))
    for i, name in enumerate(CFG.model.kws_classes):
        n = total_cls.get(i, 0)
        c = correct_cls.get(i, 0)
        print("  {:12s} {:4d}/{:4d}  = {:6.1f}%".format(name, c, n, 100 * c / n if n else 0.0))

# ── HELD-OUT TEST accuracy (speaker-disjoint) — the honest headline number ────
if test_loader is not None:
    from collections import Counter
    t_pred, t_true = [], []
    t_vad_correct = t_vad_total = 0
    with torch.no_grad():
        for feat, mask_tgt, vad_tgt, kws_tgt in test_loader:
            feat = feat.to(device, non_blocking=True)
            preds = model(feat)
            t_pred.extend(preds[2].argmax(dim=1).cpu().tolist())
            t_true.extend(kws_tgt.tolist())
            # VAD frame accuracy (is-speech detection) at 0.5 threshold
            vad_p = (preds[1].squeeze(-1).cpu() > 0.5).float()
            t_vad_correct += (vad_p == vad_tgt).sum().item()
            t_vad_total   += vad_tgt.numel()
    cc, tc = Counter(), Counter()
    for p, t in zip(t_pred, t_true):
        tc[t] += 1
        if p == t:
            cc[t] += 1
    acc = sum(cc.values()) / max(sum(tc.values()), 1)
    print("\n" + "=" * 60)
    print("HELD-OUT TEST (speaker-disjoint)  KWS overall = {:.1f}%".format(acc * 100))
    for i, name in enumerate(CFG.model.kws_classes):
        n = tc.get(i, 0); c = cc.get(i, 0)
        print("  {:12s} {:4d}/{:4d}  = {:6.1f}%".format(name, c, n, 100 * c / n if n else 0.0))
    print("VAD frame accuracy on test = {:.1f}%".format(100 * t_vad_correct / max(t_vad_total, 1)))
    print("=" * 60)

# ── Save loss curve PNG ────────────────────────────────────────────────────────
os.makedirs(CFG.export_dir, exist_ok=True)
fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
for ax, task in zip(axes, ['total', 'mask', 'vad', 'kws']):
    ax.plot(history['train_' + task], label='train', color='tab:blue',   lw=2)
    ax.plot(history['val_'   + task], label='val',   color='tab:orange', lw=2)
    ax.set_title(task.upper() + ' loss')
    ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
data_label = 'real audio (LibriSpeech)' if hasattr(train_ds, 'df') else 'synthetic data'
fig.suptitle('LiteVoiceNet loss curves ({})'.format(data_label), fontsize=13)
plt.tight_layout()
curve_path = os.path.join(CFG.export_dir, 'loss_curves.png')
plt.savefig(curve_path, dpi=120, bbox_inches='tight')
plt.close()
print("Loss curve saved -> {}".format(curve_path))

# ── ONNX export ───────────────────────────────────────────────────────────────
print("\n=== ONNX export ===")
model_cpu = model.cpu().eval()
onnx_path = os.path.join(CFG.export_dir, 'litevoicenet.onnx')
exp_mod.export_onnx(model_cpu, CFG, output_path=onnx_path, seq_frames=CFG.synth.seq_frames)

# ORT sanity check
import onnxruntime as ort
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess = ort.InferenceSession(onnx_path, sess_options=sess_opts,
                             providers=['CPUExecutionProvider'])
rng_t  = np.random.default_rng(0)
x_test = rng_t.random((1, CFG.array.num_features,
                        CFG.synth.seq_frames, CFG.signal.n_freq_bins)).astype(np.float32)
hx_t   = np.zeros((CFG.model.gru_layers, 1, CFG.model.gru_hidden), np.float32)
outs   = sess.run(None, {'features': x_test, 'h_in': hx_t})
print("ORT sanity check OK  mask={} vad={} kws={}".format(
    outs[0].shape, outs[1].shape, outs[2].shape))

# ── ONNX INT8 quantisation ────────────────────────────────────────────────────
print("\n=== INT8 quantisation ===")
int8_path = os.path.join(CFG.export_dir, 'litevoicenet_int8.onnx')
ok = exp_mod.quantize_onnx_int8(onnx_path, output_path=int8_path,
                                  cfg=CFG, calib_features=calib_features)

# ── CPU latency benchmark ─────────────────────────────────────────────────────
print("\n=== CPU latency benchmark (1 thread, simulates Pi 4) ===")
chunk_ms = CFG.synth.seq_frames * CFG.signal.hop_size / CFG.signal.sample_rate_raw * 1000
print("Audio chunk size: {:.2f} ms".format(chunk_ms))

stats32 = exp_mod.benchmark_cpu(onnx_path, CFG, n_runs=200, n_warmup=20)
print("FP32  mean={:.2f} ms  p95={:.2f} ms  RT={:.3f}x".format(
    stats32['mean_ms'], stats32['p95_ms'], stats32['realtime_factor']))

if ok:
    stats8 = exp_mod.benchmark_cpu(int8_path, CFG, n_runs=200, n_warmup=20)
    speedup = stats32['mean_ms'] / stats8['mean_ms']
    print("INT8  mean={:.2f} ms  p95={:.2f} ms  speedup={:.2f}x".format(
        stats8['mean_ms'], stats8['p95_ms'], speedup))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Export summary ===")
for name, path in [('ONNX FP32', onnx_path), ('ONNX INT8', int8_path)]:
    if os.path.exists(path):
        print("  {:<12} : {}  ({:.2f} MB)".format(
            name, path, os.path.getsize(path) / 1e6))
print("  Loss curve  : {}".format(curve_path))
print("\nDONE")
