"""Generate result figures for the thesis from the final model: confusion matrix,
per-class accuracy bar, and copy the loss curves. Outputs to docs/thesis/Figures/."""
import sys, os, shutil
sys.path.insert(0, '..'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')

if __name__ == "__main__":
    import numpy as np, torch
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from collections import Counter
    from config import CFG
    from model import build_model
    from data import ManifestDataset

    FIG = os.path.normpath(os.path.join('..', 'docs', 'thesis', 'Figures'))
    names = CFG.model.kws_classes
    device = torch.device('cuda')
    model = build_model(CFG).to(device)
    ck = torch.load(os.path.join(CFG.export_dir, 'litevoicenet_best.pt'), map_location=device)
    model.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    model.eval()

    test_ds = ManifestDataset('data/manifests/test.csv', CFG, augment=False)
    dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    n = len(names)
    conf = np.zeros((n, n), int)
    vad_c = vad_t = 0
    vad_tp = vad_tn = vad_fp = vad_fn = 0
    with torch.no_grad():
        for feat, mask_t, vad_tg, kws_t in dl:
            feat = torch.as_tensor(feat).to(device)
            preds = model(feat)
            p = preds[2].argmax(1).cpu().numpy()
            for t, pr in zip(kws_t.numpy(), p):
                conf[int(t), int(pr)] += 1
            vp = (preds[1].squeeze(-1).cpu() > 0.5).float()
            vt = torch.as_tensor(vad_tg).float()
            vad_c += (vp == vt).sum().item(); vad_t += vad_tg.numel()
            vad_tp += int(((vp == 1) & (vt == 1)).sum()); vad_fn += int(((vp == 0) & (vt == 1)).sum())
            vad_tn += int(((vp == 0) & (vt == 0)).sum()); vad_fp += int(((vp == 1) & (vt == 0)).sum())
    acc = conf.diagonal().sum() / conf.sum()
    vad_recall = 100 * vad_tp / max(vad_tp + vad_fn, 1)   # voiced frames detected
    vad_spec = 100 * vad_tn / max(vad_tn + vad_fp, 1)     # non-vocal frames rejected
    print("overall test KWS acc {:.1f}%  VAD {:.1f}%".format(100*acc, 100*vad_c/vad_t))
    print("VAD voice-recall {:.1f}%  non-vocal-rejection {:.1f}%".format(vad_recall, vad_spec))

    # confusion matrix (row-normalized)
    cn = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_title('KWS confusion matrix (held-out test)')
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "{:.0f}%".format(100*cn[i, j]), ha='center', va='center',
                    color='white' if cn[i, j] > 0.5 else 'black', fontsize=9)
    plt.colorbar(im, fraction=0.046); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'confusion_matrix.png'), dpi=130); plt.close()
    print("saved confusion_matrix.png")

    # per-class accuracy bar
    per = conf.diagonal() / conf.sum(1)
    plt.figure(figsize=(6, 3.4))
    colors = ['tab:green' if names[i] in ('help', 'stop') else 'tab:blue' for i in range(n)]
    plt.bar(names, per * 100, color=colors)
    plt.axhline(100/n, ls='--', color='gray', label='chance')
    plt.ylabel('accuracy (%)'); plt.ylim(0, 100)
    plt.title('Per-class accuracy on held-out test\n(VAD: voice recall {:.0f}%, non-vocal rejection {:.0f}%)'.format(vad_recall, vad_spec))
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'per_class_accuracy.png'), dpi=130); plt.close()
    print("saved per_class_accuracy.png")

    if os.path.exists(os.path.join(CFG.export_dir, 'loss_curves.png')):
        shutil.copy(os.path.join(CFG.export_dir, 'loss_curves.png'),
                    os.path.join(FIG, 'loss_curves.png'))
        print("copied loss_curves.png")

    # print confusion for the write-up
    print("\nconfusion (rows=true):")
    print("        " + "  ".join("{:>7s}".format(x) for x in names))
    for i in range(n):
        print("  {:7s} ".format(names[i]) + "  ".join("{:7d}".format(conf[i, j]) for j in range(n)))
