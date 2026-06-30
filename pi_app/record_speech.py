"""
Inregistreaza propozitii generale prin UMA-8 pentru clasa 'speech'.

Scriptul iti arata o propozitie, astepti 1s, apoi inregistreaza 3s,
pauza 0.5s, urmatoarea. Clipurile se salveaza in:
  training/data/speech_real/

Dupa ce termini, ruleaza:
  python training/add_speech_to_manifest.py

Utilizare:
  python pi_app/record_speech.py [--device 25] [--count 50]
"""

import argparse
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

SENTENCES = [
    "Buna ziua, cum va mai mergeti astazi?",
    "Ma duc la cumparaturi, aveti nevoie de ceva?",
    "Vremea este frumoasa afara astazi",
    "Pot sa va ajut cu ceva?",
    "Multumesc mult pentru ajutor",
    "La revedere si o zi buna",
    "Ce mai faci? Totul bine?",
    "Am terminat treaba pentru azi",
    "Sa mergem la o cafea dupa aceea",
    "Ora este trei si jumatate",
    "Astazi am o intalnire importanta",
    "Imi place muzica clasica",
    "Sistemul functioneaza foarte bine",
    "Computerul este pe birou",
    "Fac o pauza de cinci minute",
    "Imi este foame, ce mancam?",
    "Drona zboara frumos",
    "Inregistrarea a inceput deja",
    "Testam microfoanele acum",
    "Vocea mea suna natural?",
    "Temperatura exterioara este douazeci de grade",
    "Am nevoie de o carte buna",
    "Proiectul de licenta merge bine",
    "Saptamana viitoare am examenul final",
    "Multumesc pentru rabdare",
    "Sa continuam cu urmatorul pas",
    "Sistemul de recunoastere vocala functioneaza",
    "Matricea de microfoane detecteaza sunetele",
    "Procesarea semnalului audio este complexa",
    "Algoritmul de localizare este precis",
    "Ma pregatesc pentru prezentarea finala",
    "Colegii mei au proiecte interesante",
    "Facultatea de automatica si calculatoare",
    "Ingineria sistemelor de calcul este fascinanta",
    "Am invatat multe lucruri noi in acest an",
    "Proiectul implica retele neuronale",
    "Datele de antrenament sunt importante",
    "Modelul are douazeci si opt de mii de parametri",
    "Acuratetea clasificarii este buna",
    "Latenta sistemului este de douazeci de milisecunde",
    "Raspberry Pi este o placa de dezvoltare",
    "CUDA accelereaza antrenamentul retelei",
    "Augmentarea datelor imbunatateste performanta",
    "Recunoasterea cuvintelor cheie este dificila",
    "Microfoanele MEMS au sensibilitate buna",
    "Formatele ONNX sunt eficiente pentru inferenta",
    "Cuantizarea INT8 reduce dimensiunea modelului",
    "Directia de sosire se calculeaza cu SRP-PHAT",
    "Filtrul Wiener reduce zgomotul de fond",
    "Reteaua neuronala are trei capete de iesire",
]

SR          = 48_000
DURATION_S  = 3.0
PAUSE_S     = 0.8
WARMUP_S    = 0.8
CH          = 8
REF_MIC     = 0


def record_clip(device, duration=DURATION_S, sr=SR):
    samples = int(duration * sr)
    audio = sd.rec(samples, samplerate=sr, channels=CH, device=device,
                   dtype="float32", blocking=True)
    return audio[:, REF_MIC]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=25)
    ap.add_argument("--count",  type=int, default=50,
                    help="Cate propozitii sa inregistrezi (max 50)")
    args = ap.parse_args()

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "training", "data", "speech_real"
    )
    os.makedirs(out_dir, exist_ok=True)

    sentences = SENTENCES[:args.count]
    print(f"\nInregistrare speech prin UMA-8 (device={args.device})")
    print(f"Clipuri: {len(sentences)} x {DURATION_S}s = {len(sentences)*DURATION_S:.0f}s total")
    print(f"Salvare in: {out_dir}")
    print("\nInstructiuni:")
    print("  - Citeste propozitia aratata si vorbeste natural")
    print("  - Nu e nevoie sa o zici perfect, vocea normala e mai buna")
    print("  - Astepti ca READY sa apara, apoi incepi sa vorbesti\n")
    input("Apasa ENTER pentru a incepe...")

    saved = 0
    for i, sentence in enumerate(sentences):
        print(f"\n[{i+1:02d}/{len(sentences)}] {sentence}")
        print("  Pregateste-te...", end="", flush=True)
        time.sleep(WARMUP_S)
        print(" READY - VORBESTE")

        clip = record_clip(args.device)
        r = rms(clip)

        if r < 0.0004:
            print(f"  [skip] silentiu complet (rms={r:.4f}) - apasa ENTER sa reincerci sau S+ENTER sa sari")
            c = input("  > ").strip().lower()
            if c == "s":
                continue
            print("  READY (reinregistrare) - VORBESTE acum")
            clip = record_clip(args.device)
            r = rms(clip)

        fname = f"speech_real_{saved:04d}.wav"
        fpath = os.path.join(out_dir, fname)
        peak = float(np.abs(clip).max())
        if peak > 0.95:
            clip = clip * (0.95 / peak)
        sf.write(fpath, clip, SR)
        saved += 1
        print(f"  Salvat: {fname}  (rms={r:.4f})")

        time.sleep(PAUSE_S)

    print(f"\nGata! {saved} clipuri salvate in {out_dir}")
    print("Acum ruleaza:")
    print("  python training/add_speech_to_manifest.py")


if __name__ == "__main__":
    main()
