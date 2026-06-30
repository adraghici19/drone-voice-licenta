"""Generate explanatory figures for the middle chapters (Ch3-Ch4) of the thesis:
  - array_geometry.png    : real UMA-8 v2 mic layout (1 central + 6 peripheral)
  - beam_pattern.png      : delay-and-sum directivity of the 9 cm array vs frequency
  - litevoicenet_arch.png : LiteVoiceNet block diagram (matches model.py exactly)
  - feature_channels.png  : the 5-channel network input on a real synthetic sample
Outputs to docs/thesis/Figures/.  Everything is computed from the real code/geometry.
"""
import sys, os
sys.path.insert(0, '..'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.normpath(os.path.join('..', 'docs', 'thesis', 'Figures'))
os.makedirs(FIG, exist_ok=True)

from shared.array_geometry import MIC_POSITIONS, ARRAY_RADIUS_M, SPEED_OF_SOUND_MS

# ───────────────────────────── 1. Array geometry ─────────────────────────────
def fig_geometry():
    R = ARRAY_RADIUS_M * 100.0  # cm
    # Peripheral-mic angles matching the real UMA-8 v2 silkscreen:
    # MIC1 at the top (12 o'clock), numbered CLOCKWISE 1->6.
    ang = {1: 90.0, 2: 30.0, 3: -30.0, 4: -90.0, 5: 210.0, 6: 150.0}
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.add_patch(plt.Circle((0, 0), R, fill=False, ls='--', color='gray', lw=1))
    # central mic
    ax.scatter([0], [0], s=190, c='tab:orange', zorder=3, edgecolors='k')
    ax.annotate('0 (centre)', (0, 0), textcoords='offset points', xytext=(10, 8),
                fontsize=10, fontweight='bold')
    # peripheral mics
    for m, a in ang.items():
        x, y = R * np.cos(np.deg2rad(a)), R * np.sin(np.deg2rad(a))
        ax.scatter([x], [y], s=190, c='tab:blue', zorder=3, edgecolors='k')
        ox = 12 * np.sign(x) if abs(x) > 0.1 else 0
        oy = 12 * np.sign(y) if abs(y) > 0.1 else 12
        ax.annotate(str(m), (x, y), textcoords='offset points', xytext=(ox, oy),
                    fontsize=11, fontweight='bold', ha='center', va='center')
    # 0-degree reference arrow (+x), label kept well inside the axes
    ax.annotate('', xy=(R * 1.22, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.6))
    ax.text(R * 0.62, R * 0.30, r'$0^\circ$ reference', color='tab:red',
            fontsize=9.5, ha='left', va='bottom')
    # diameter annotation
    ax.annotate('', xy=(-R, -R * 1.35), xytext=(R, -R * 1.35),
                arrowprops=dict(arrowstyle='<->', color='dimgray'))
    ax.text(0, -R * 1.58, r'$\approx${:.0f} cm'.format(2 * R),
            ha='center', color='dimgray', fontsize=10)
    lim = R * 1.85
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal'); ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
    ax.set_title('UMA-8 v2 layout: 1 central + 6 peripheral MEMS')
    ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'array_geometry.png'), dpi=140); plt.close()
    print('saved array_geometry.png')

# ───────────────────────── 2. Delay-and-sum beam pattern ──────────────────────
def fig_beampattern():
    peri = MIC_POSITIONS[1:7]                 # 6 peripheral mics
    c = SPEED_OF_SOUND_MS
    th = np.deg2rad(np.linspace(-180, 180, 721))
    th0 = 0.0                                 # steer to 0 deg
    u0 = np.array([np.cos(th0), np.sin(th0)])
    fig = plt.figure(figsize=(5.0, 4.8))
    ax = plt.subplot(111, projection='polar')
    for f in [500, 1000, 2000, 4000]:
        k = 2 * np.pi * f / c
        B = np.zeros_like(th, dtype=complex)
        for a in range(len(th)):
            u = np.array([np.cos(th[a]), np.sin(th[a])])
            B[a] = np.mean(np.exp(1j * k * (peri @ (u - u0))))
        mag = 20 * np.log10(np.abs(B) + 1e-6)
        mag = np.clip(mag, -25, 0)
        ax.plot(th, mag + 25, label='{} Hz'.format(f), lw=1.6)
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    ax.set_rticks([5, 15, 25]); ax.set_yticklabels(['-20', '-10', '0 dB'])
    ax.set_rlabel_position(135)
    ax.set_title('Delay-and-sum directivity (steered to $0^\\circ$)', pad=18)
    ax.legend(loc='lower right', bbox_to_anchor=(1.18, -0.05), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'beam_pattern.png'), dpi=140); plt.close()
    print('saved beam_pattern.png')

# ──────────────────────── 3. LiteVoiceNet architecture ────────────────────────
def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04',
                                fc=fc, ec='k', lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.4)

def _arrow(ax, x0, y0, x1, y1, color='k', style='-|>'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=12, color=color, lw=1.1))

def fig_arch():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')
    blue, green, orange, grey = '#cfe2ff', '#d7f0d2', '#ffe2c4', '#e9e9e9'
    _box(ax, 0.1, 2.0, 1.35, 1.0,
         'Input\n5 ch x 100 x 257\n(1 log-mag + 4 IPD)', grey)
    _box(ax, 1.75, 2.0, 1.75, 1.0,
         'Encoder\n4x DS-Conv (causal)\nch 5->32->64->96->128\nF 257->129->65->33->17', blue)
    _box(ax, 3.8, 2.0, 1.5, 1.0, '2x GRU\n(unidirectional)\nhidden 128', blue)
    # heads
    _box(ax, 5.9, 3.45, 3.0, 0.9, 'VAD head: Linear -> sigmoid\nper-frame voice activity', green)
    _box(ax, 5.9, 2.05, 3.0, 0.9, 'KWS head: avg-pool + MLP\n4 classes (help/stop/speech/noise)', green)
    _box(ax, 5.9, 0.55, 3.0, 0.95, 'Mask decoder: U-Net (4x transpose)\nF 17->257 -> sigmoid IRM mask', orange)
    # main flow
    _arrow(ax, 1.45, 2.5, 1.75, 2.5)
    _arrow(ax, 3.5, 2.5, 3.8, 2.5)
    _arrow(ax, 5.3, 2.5, 5.6, 2.5)
    _arrow(ax, 5.6, 2.5, 5.9, 3.9)   # to VAD
    _arrow(ax, 5.6, 2.5, 5.9, 2.5)   # to KWS
    _arrow(ax, 5.6, 2.5, 5.9, 1.0)   # to mask
    # skip connections (encoder -> mask decoder), curved under the main flow
    ax.add_patch(FancyArrowPatch((2.55, 1.95), (5.85, 1.10), arrowstyle='-|>',
                                 mutation_scale=12, color='tab:red', lw=1.1,
                                 connectionstyle='arc3,rad=0.22'))
    ax.text(4.15, 0.40, 'U-Net skip connections', color='tab:red', fontsize=7.5,
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=0.4))
    ax.text(5.0, 4.55, 'LiteVoiceNet — 278,019 parameters, causal, 1.1 MB ONNX (0.9 MB INT8)',
            ha='center', fontsize=9.6, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'litevoicenet_arch.png'), dpi=150); plt.close()
    print('saved litevoicenet_arch.png')

# ──────────────────────── 4. 5-channel feature example ────────────────────────
def fig_features():
    from config import CFG
    from data import SyntheticAudioDataset
    ds = SyntheticAudioDataset(8, CFG, base_seed=7)
    feat, irm, vad, kws = ds[0]               # feat: (5, T, F)
    feat = np.asarray(feat)
    titles = ['log-magnitude (ref. mic)',
              'IPD pair 1 — sin', 'IPD pair 1 — cos',
              'IPD pair 2 — sin', 'IPD pair 2 — cos']
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7))
    for k in range(5):
        a = axes[k]
        im = a.imshow(feat[k].T, origin='lower', aspect='auto', cmap='magma')
        a.set_title(titles[k], fontsize=8.5)
        a.set_xlabel('frame'); a.set_yticks([])
        if k == 0:
            a.set_ylabel('frequency bin')
    fig.suptitle('Five-channel network input on a speech-in-noise example', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIG, 'feature_channels.png'), dpi=140); plt.close()
    print('saved feature_channels.png')

if __name__ == '__main__':
    fig_geometry()
    fig_beampattern()
    fig_arch()
    try:
        fig_features()
    except Exception as e:
        print('feature_channels SKIPPED:', repr(e))
    print('done')
