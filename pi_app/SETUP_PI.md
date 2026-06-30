# Raspberry Pi 4 — headless (SSH) setup for LiteVoiceNet

Goal: a Pi 4 with no monitor, reachable over your network by SSH, running the
UMA-8 + ONNX inference. You configure everything from your PC before first boot.

Recommended OS: **Raspberry Pi OS Lite (64-bit)** — no desktop, lighter on the
4 GB Pi 4, and it matches the embedded-deployment story for the thesis.

---

## Part A — Flash the SD card (do this now, in parallel with training)

1. **Install Raspberry Pi Imager** on your Windows PC:
   https://www.raspberrypi.com/software/  → "Download for Windows" → install.

2. Insert the SD card into your PC (use the USB adapter if needed).

3. Open Raspberry Pi Imager:
   - **Choose Device**: Raspberry Pi 4
   - **Choose OS**: "Raspberry Pi OS (other)" → **Raspberry Pi OS Lite (64-bit)**
   - **Choose Storage**: your SD card (⚠ double-check it's the card, not a USB drive)

4. Click **Next**, then **"Edit Settings"** (the headless magic — do NOT skip):
   - **General tab**:
     - Set hostname: `dronepi`  (so you can reach it at `dronepi.local`)
     - Set username + password: e.g. user `pi`, a password you'll remember
     - Configure wireless LAN: your WiFi **SSID** + **password** + Wireless LAN
       country `RO`
     - Set locale / timezone: `Europe/Bucharest`
   - **Services tab**:
     - ✅ **Enable SSH** → "Use password authentication"
   - Save.

5. Click **Yes** to apply settings, **Yes** to erase + write. Wait ~5–10 min.

6. When done, eject the card, put it in the Pi.

> ⚠ Use the **same WiFi network your PC is on**, otherwise `.local` discovery and
> SSH won't reach it.

---

## Part B — First boot + SSH in

1. Insert the SD card into the Pi, plug in the **UMA-8 via USB**, then power the Pi.
   First boot takes ~1–2 minutes (it expands the filesystem and joins WiFi).

2. From your PC's PowerShell, connect:
   ```powershell
   ssh pi@dronepi.local
   ```
   - First time it asks to trust the host → type `yes`.
   - Enter the password you set in the Imager.
   - If `dronepi.local` doesn't resolve, find the Pi's IP from your router's
     device list (or `ping dronepi.local`) and use `ssh pi@<IP>` instead.

3. You're in. Quick check it sees the array:
   ```bash
   arecord -l        # should list "miniDSP" / "UMA-8" as a card
   ```

---

## Part C — Install dependencies (on the Pi, over SSH)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libportaudio2 libsndfile1
mkdir -p ~/drone && cd ~/drone
python3 -m venv .venv
source .venv/bin/activate
pip install numpy sounddevice soundfile onnxruntime
```

If `pip install onnxruntime` can't find a wheel:
```bash
pip install --index-url https://www.piwheels.org/simple onnxruntime
```

---

## Part D — Copy the app + model from your PC to the Pi

Run these **on your PC** (PowerShell), from the project root. `scp` ships with
Windows 10/11.

```powershell
# the inference app (no PyTorch needed)
scp -r pi_app pi@dronepi.local:/home/pi/drone/

# the exported models
scp -r training/exported pi@dronepi.local:/home/pi/drone/training-exported
```

The app finds the model via the relative path `../training/exported/...`. On the
Pi we copied it to `training-exported`, so set the path once on the Pi:

```bash
cd ~/drone/pi_app
# point config at the copied models (edit onnx_fp32 / onnx_int8 paths), or simply:
ln -s /home/pi/drone/training-exported /home/pi/drone/training/exported  # after: mkdir -p ~/drone/training
```
(We'll do this step together — it's a one-line path fix.)

---

## Part E — Find the UMA-8 index and run

```bash
cd ~/drone/pi_app
source ~/drone/.venv/bin/activate
python check_channels.py --list        # find the UMA-8 device index on the Pi
python check_channels.py --device <idx> --seconds 5   # verify the 7 mics
python main.py --device <idx>          # live keyword spotting!
```

The device index on the Pi will differ from the PC's (25). On Linux/ALSA it's
usually small (e.g. 1 or 2). `first_mic_channel` should stay `0` (same firmware).

---

## What you can do NOW (in parallel with training)

**Part A only** — flash the SD card and do the headless config. Stop there.
We'll do Parts B–E together once the model finishes training and passes the
live test on the PC. Tell me when the card is flashed (or if Imager asks
something you're unsure about).
