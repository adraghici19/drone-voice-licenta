"""
Live dashboard for LiteVoiceNet on Raspberry Pi.
Run:  python3 dashboard.py [--device N] [--threshold 0.25] [--port 5000]
Open: http://<pi-ip>:5000  in any browser on the same network.
"""

import argparse
import threading
import time
import sys
import numpy as np
from flask import Flask, jsonify, render_template_string

from config import PI_CFG
from features import build_features
from infer import LiteVoiceNetONNX
from detect import KeywordDetector
from doa import SrpPhatDoA, hex_mic_positions

_ACTIVE_MICS = [0, 1, 2, 3, 4, 6]

# ── shared state (written by audio thread, read by Flask) ─────────────────────
_state = {
    "vad": False,
    "scores": {"help": 0.0, "stop": 0.0, "speech": 0.0, "noise": 1.0},
    "direction": None,
    "event": None,
    "event_time": 0.0,
    "log": [],          # last 8 events
    "running": False,
    "error": None,
}
_lock = threading.Lock()

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LiteVoiceNet — Live Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d2137;
    color: #d6e8ff;
    font-family: 'Segoe UI', sans-serif;
    min-height: 100vh;
    padding: 18px;
  }
  h1 {
    font-size: 1.25rem;
    font-weight: 400;
    color: #fff;
    letter-spacing: 1px;
    border-bottom: 2px solid #4caf50;
    padding-bottom: 8px;
    margin-bottom: 18px;
  }
  h1 span { color: #4caf50; font-weight: 700; }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    max-width: 820px;
    margin: 0 auto;
  }
  @media(max-width:540px){ .grid { grid-template-columns: 1fr; } }

  .card {
    background: #071426;
    border-radius: 10px;
    padding: 16px 18px;
    border: 1px solid #1a3a5c;
  }
  .card-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4caf50;
    margin-bottom: 10px;
  }

  /* VAD indicator */
  #vad-box {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  #vad-led {
    width: 52px; height: 52px;
    border-radius: 50%;
    background: #1a3a1a;
    border: 3px solid #2a5a2a;
    transition: background .15s, border-color .15s, box-shadow .15s;
  }
  #vad-led.active {
    background: #4caf50;
    border-color: #81e07a;
    box-shadow: 0 0 18px #4caf5088;
  }
  #vad-label {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a3a1a;
    transition: color .15s;
  }
  #vad-label.active { color: #4caf50; }

  /* KWS bars */
  .bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 7px;
  }
  .bar-label {
    width: 52px;
    font-size: 0.82rem;
    color: #88bbff;
    text-align: right;
    flex-shrink: 0;
  }
  .bar-track {
    flex: 1;
    background: #0f2742;
    border-radius: 4px;
    height: 18px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width .18s;
    min-width: 0;
  }
  .bar-val {
    width: 38px;
    font-size: 0.82rem;
    color: #88bbff;
    text-align: left;
    flex-shrink: 0;
  }
  .bar-help  { background: #4caf50; }
  .bar-stop  { background: #00bcd4; }
  .bar-speech{ background: #f59a00; }
  .bar-noise { background: #555e6a; }

  /* Compass */
  #compass-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  canvas#compass {
    border-radius: 50%;
  }
  #direction-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00bcd4;
    letter-spacing: 1px;
  }
  #direction-val.inactive { color: #1a3a5c; }

  /* Event flash */
  #event-box {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 64px;
    border-radius: 8px;
    background: #071426;
    border: 2px solid #1a3a5c;
    transition: background .2s, border-color .2s;
  }
  #event-box.fired {
    background: #1a4a1a;
    border-color: #4caf50;
  }
  #event-text {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a3a5c;
    letter-spacing: 3px;
    transition: color .2s;
  }
  #event-text.fired { color: #4caf50; }

  /* Log */
  #log-list {
    list-style: none;
    font-size: 0.8rem;
    line-height: 1.9;
    color: #5588aa;
    max-height: 148px;
    overflow-y: auto;
  }
  #log-list li:first-child { color: #88ddaa; }

  /* Status bar */
  #status {
    max-width: 820px;
    margin: 10px auto 0;
    font-size: 0.72rem;
    color: #2a4a6a;
    text-align: right;
  }
  #status.err { color: #ff6666; }
</style>
</head>
<body>
<h1>LiteVoiceNet &nbsp;<span>LIVE</span></h1>

<div class="grid">

  <!-- VAD -->
  <div class="card">
    <div class="card-title">Voice Activity</div>
    <div id="vad-box">
      <div id="vad-led"></div>
      <div id="vad-label">SILENCE</div>
    </div>
  </div>

  <!-- Event flash -->
  <div class="card">
    <div class="card-title">Detected Keyword</div>
    <div id="event-box">
      <div id="event-text">—</div>
    </div>
  </div>

  <!-- KWS bars -->
  <div class="card">
    <div class="card-title">Class Probabilities</div>
    <div id="bars"></div>
  </div>

  <!-- Compass -->
  <div class="card">
    <div class="card-title">Direction of Arrival</div>
    <div id="compass-wrap">
      <canvas id="compass" width="130" height="130"></canvas>
      <div id="direction-val" class="inactive">— °</div>
    </div>
  </div>

  <!-- Log -->
  <div class="card" style="grid-column:1/-1">
    <div class="card-title">Event Log</div>
    <ul id="log-list"><li>waiting for events…</li></ul>
  </div>

</div>

<div id="status">connecting…</div>

<script>
const CLASSES = ['help','stop','speech','noise'];
const COLORS  = ['bar-help','bar-stop','bar-speech','bar-noise'];

// build bar rows once
const barsEl = document.getElementById('bars');
CLASSES.forEach((c,i) => {
  barsEl.innerHTML += `
    <div class="bar-row">
      <span class="bar-label">${c}</span>
      <div class="bar-track"><div id="bar-${c}" class="bar-fill ${COLORS[i]}" style="width:0%"></div></div>
      <span class="bar-val" id="val-${c}">0.00</span>
    </div>`;
});

// compass canvas
const cv = document.getElementById('compass');
const ctx = cv.getContext('2d');
let lastAz = null;

function drawCompass(az) {
  const W = cv.width, H = cv.height, cx = W/2, cy = H/2, R = W/2 - 6;
  ctx.clearRect(0,0,W,H);

  // background circle
  ctx.beginPath(); ctx.arc(cx,cy,R,0,2*Math.PI);
  ctx.fillStyle = '#0f2742'; ctx.fill();
  ctx.strokeStyle = '#1a3a5c'; ctx.lineWidth = 2; ctx.stroke();

  // cardinal labels
  ctx.fillStyle = '#4a7aa0'; ctx.font = '11px Segoe UI'; ctx.textAlign = 'center';
  ctx.fillText('N', cx,      cy-R+14);
  ctx.fillText('S', cx,      cy+R-4);
  ctx.fillText('E', cx+R-8,  cy+4);
  ctx.fillText('W', cx-R+8,  cy+4);

  if (az === null) {
    ctx.beginPath(); ctx.arc(cx,cy,5,0,2*Math.PI);
    ctx.fillStyle='#1a3a5c'; ctx.fill();
    return;
  }

  // arrow
  const rad = (az - 90) * Math.PI / 180;
  const tip = {x: cx + (R-10)*Math.cos(rad), y: cy + (R-10)*Math.sin(rad)};
  const tail = {x: cx - 22*Math.cos(rad),    y: cy - 22*Math.sin(rad)};

  ctx.beginPath();
  ctx.moveTo(tail.x, tail.y);
  ctx.lineTo(tip.x, tip.y);
  ctx.strokeStyle = '#00bcd4'; ctx.lineWidth = 3;
  ctx.lineCap = 'round'; ctx.stroke();

  // arrowhead
  const a1 = rad + 2.5, a2 = rad - 2.5;
  ctx.beginPath();
  ctx.moveTo(tip.x, tip.y);
  ctx.lineTo(tip.x - 10*Math.cos(a1), tip.y - 10*Math.sin(a1));
  ctx.lineTo(tip.x - 10*Math.cos(a2), tip.y - 10*Math.sin(a2));
  ctx.closePath();
  ctx.fillStyle = '#00bcd4'; ctx.fill();

  // center dot
  ctx.beginPath(); ctx.arc(cx,cy,4,0,2*Math.PI);
  ctx.fillStyle='#fff'; ctx.fill();
}
drawCompass(null);

let lastEventKw = null;
let eventTimer = null;

function update() {
  fetch('/api/state')
    .then(r => r.json())
    .then(d => {
      // VAD
      const led   = document.getElementById('vad-led');
      const vlbl  = document.getElementById('vad-label');
      if (d.vad) { led.classList.add('active'); vlbl.classList.add('active'); vlbl.textContent = 'SPEECH'; }
      else        { led.classList.remove('active'); vlbl.classList.remove('active'); vlbl.textContent = 'SILENCE'; }

      // KWS bars
      CLASSES.forEach(c => {
        const p = d.scores[c] || 0;
        document.getElementById('bar-'+c).style.width = (p*100).toFixed(1)+'%';
        document.getElementById('val-'+c).textContent  = p.toFixed(2);
      });

      // Direction
      const dv = document.getElementById('direction-val');
      if (d.direction !== null && d.direction !== undefined) {
        drawCompass(d.direction);
        dv.textContent = d.direction.toFixed(1) + ' °';
        dv.classList.remove('inactive');
      } else {
        drawCompass(null);
        dv.textContent = '— °';
        dv.classList.add('inactive');
      }

      // Event flash
      if (d.event && d.event !== lastEventKw) {
        lastEventKw = d.event;
        const eb  = document.getElementById('event-box');
        const et  = document.getElementById('event-text');
        eb.classList.add('fired'); et.classList.add('fired');
        et.textContent = d.event.toUpperCase();
        clearTimeout(eventTimer);
        eventTimer = setTimeout(() => {
          eb.classList.remove('fired'); et.classList.remove('fired');
          et.textContent = '—'; lastEventKw = null;
        }, 2200);
      }

      // Log
      if (d.log && d.log.length) {
        const ul = document.getElementById('log-list');
        ul.innerHTML = d.log.map((e,i) => `<li>${e}</li>`).join('');
      }

      // Status
      const st = document.getElementById('status');
      st.textContent = d.running ? 'live · ' + new Date().toLocaleTimeString() : 'pipeline not running';
      st.className = d.error ? 'err' : '';
    })
    .catch(() => {
      document.getElementById('status').textContent = 'connection lost…';
    });
}

setInterval(update, 220);
update();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify(dict(_state))


# ── Audio + inference loop (background thread) ────────────────────────────────
def _pipeline(cfg, device):
    try:
        from audio import MicArrayStream, find_uma8
    except ImportError as e:
        with _lock:
            _state["error"] = str(e)
        return

    if device is None:
        try:
            device = find_uma8()
        except RuntimeError as e:
            with _lock:
                _state["error"] = str(e)
            return

    try:
        net      = LiteVoiceNetONNX(cfg)
        detector = KeywordDetector(cfg)
        doa      = SrpPhatDoA(cfg.sample_rate, n_fft=512,
                              mic_xy=hex_mic_positions(), grid_deg=5.0,
                              fmin=300.0, fmax=3400.0)
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
        return

    hop_period = cfg.hop_samples / cfg.sample_rate

    with _lock:
        _state["running"] = True

    with MicArrayStream(cfg, device=device) as stream:
        while not stream.has_window():
            time.sleep(0.01)

        while True:
            t0  = time.monotonic()
            raw = stream._latest(cfg.window_samples)

            needed = sorted({cfg.ref_mic} | {m for p in cfg.mic_pairs for m in p})
            rows   = max(needed) + 1
            window = np.zeros((rows, cfg.window_samples), dtype=np.float32)
            for idx in needed:
                window[idx] = raw[cfg.first_mic_channel + idx]

            feats             = build_features(window, cfg.mic_pairs, cfg.ref_mic,
                                               cfg.n_fft, cfg.hop_size)
            mask, vad, kws    = net.run(feats)
            vad_active        = float(vad.mean()) > cfg.vad_threshold
            kw                = detector.update(kws, now=time.monotonic())

            az = None
            if vad_active:
                mic6   = raw[_ACTIVE_MICS]
                az, _  = doa.estimate(mic6)

            scores = {c: float(p) for c, p in zip(cfg.kws_classes, detector.smoothed)}

            with _lock:
                _state["vad"]       = vad_active
                _state["scores"]    = scores
                _state["direction"] = round(az, 1) if az is not None else None

                if kw:
                    ts  = time.strftime("%H:%M:%S")
                    az_s = f"{az:.0f}°" if az is not None else "—"
                    entry = f"[{ts}]  {kw.upper():<6}  {az_s}"
                    _state["event"]      = kw
                    _state["event_time"] = time.monotonic()
                    log = _state["log"]
                    log.insert(0, entry)
                    _state["log"] = log[:8]
                elif time.monotonic() - _state["event_time"] > 2.5:
                    _state["event"] = None

            elapsed = time.monotonic() - t0
            sleep   = hop_period - elapsed
            if sleep > 0:
                time.sleep(sleep)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="LiteVoiceNet web dashboard")
    ap.add_argument("--device",    type=int,   default=None)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--port",      type=int,   default=5000)
    ap.add_argument("--fp32",      action="store_true")
    args = ap.parse_args()

    cfg = PI_CFG
    cfg.kws_threshold = args.threshold
    cfg.vad_threshold = max(0.15, args.threshold - 0.05)
    if args.fp32:
        cfg.use_int8 = False

    t = threading.Thread(target=_pipeline, args=(cfg, args.device), daemon=True)
    t.start()

    print(f"[dashboard] open  http://0.0.0.0:{args.port}  in your browser")
    print(f"[dashboard] threshold={args.threshold}  device={args.device}")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
