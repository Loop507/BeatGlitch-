"""
BeatGlitch – Audio-Reactive Glitching
======================================
Dipendenze:
    pip install opencv-python librosa soundfile moviepy numpy

Uso:
    python beatglitch.py --input video.mp4 --output glitch_out.mp4
    python beatglitch.py --input video.mp4 --output glitch_out.mp4 --intensity 1.4 --grit 0.9
"""

import argparse
import numpy as np
import cv2
import librosa
import soundfile as sf
import tempfile
import os
from moviepy.editor import VideoFileClip, AudioFileClip

# ──────────────────────────────────────────────────────────────────────────────
# 1. ANALISI VIDEO – energia frame-per-frame
#    Confronta ogni frame col precedente (pixel diff).
#    Output: array normalizzato [0.0 … 1.0], un valore per frame.
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video(video_path: str, res=(160, 120)) -> np.ndarray:
    """
    Calcola quanto 'cambia' ogni frame rispetto al precedente.
    - Usa una risoluzione ridotta (160×120) per velocità.
    - Score = media(diff) + std(diff): cattura sia il movimento medio
      che il 'caos' locale (glitch, flash, tagli di scena).
    - Normalizzazione su [0,1] + curva potenza 1.5 (enfatizza i picchi).
    """
    cap = cv2.VideoCapture(video_path)
    scores = []
    prev = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, res)

        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            scores.append(float(np.mean(diff) + np.std(diff)))
        else:
            scores.append(0.0)
        prev = gray

    cap.release()

    arr = np.array(scores, dtype=np.float32)
    mx = arr.max()
    if mx > 0:
        arr = arr / mx          # normalizza
        arr = np.power(arr, 1.5) # curva: enfatizza i picchi
    else:
        arr = np.full_like(arr, 0.1)  # video statico: energia piatta bassa

    return arr


# ──────────────────────────────────────────────────────────────────────────────
# 2. SINTESI AUDIO GLITCH
#    L'energia video pilota tre strati:
#      A) Sintesi granulare reattiva  – frammenti ("grani") del suono originale
#         riposizionati e distorti in base ai picchi video.
#      B) Bitcrush (grit)             – quantizzazione del segnale → "grana".
#      C) Drone 55 Hz                 – texture di basso sempre presente,
#         modulata in ampiezza dall'energia video.
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_audio(
    video_path: str,
    energy: np.ndarray,
    duration: float,
    params: dict,
    sr: int = 44100
) -> np.ndarray:
    """
    Ritorna array stereo float32 shape (2, N).

    Parametri in `params`:
        v_orig_vol  – volume dell'audio originale nel mix finale (0..1)
        v_mix       – scala l'intensità dello strato glitch granulare (0..5)
        grit        – quantizzazione bitcrush: 0=nessuna, 1=massima (0..1)
        g_size      – durata massima di un grano in secondi (0.01..0.5)
        intensity   – moltiplica l'energia video → più o meno glitch (0..2)
        drone_vol   – volume del drone 55 Hz (0..1)
        seed        – seed numpy per riproducibilità
    """
    np.random.seed(int(params["seed"]))
    N = int(duration * sr)

    # ── Caricamento audio sorgente dal video ──────────────────────────────────
    try:
        src, _ = librosa.load(video_path, sr=sr, mono=False)
        if src.ndim == 1:
            src = np.tile(src, (2, 1))          # mono → stereo duplicato
        # adatta lunghezza
        if src.shape[1] < N:
            src = np.pad(src, ((0, 0), (0, N - src.shape[1])))
        else:
            src = src[:, :N]
    except Exception:
        src = np.zeros((2, N), dtype=np.float32)

    # ── Mappa energia (frame) → campioni audio ────────────────────────────────
    e_map = np.interp(
        np.linspace(0, 1, N),
        np.linspace(0, 1, len(energy)),
        energy
    ).astype(np.float32)

    # ── Strato A: sintesi granulare ───────────────────────────────────────────
    glitch_layer = np.zeros((2, N), dtype=np.float32)
    step = 0.005        # passo temporale tra grani: 5 ms
    g_size_max = params["g_size"]
    v_mix = params["v_mix"]
    grit = params["grit"]
    intensity = params["intensity"]

    for t in np.arange(0, duration - 0.05, step):
        i = int(t * sr)
        power = e_map[i] * intensity

        # Attiva il grano se c'è abbastanza energia OPPURE con probabilità 5%
        # (il 5% garantisce un minimo di glitch anche in zone quiete)
        if power <= 0.02 and np.random.random() >= 0.05:
            continue

        g_len = int(sr * np.random.uniform(0.01, g_size_max))
        if i + g_len >= N:
            continue

        grain = src[:, i : i + g_len].copy()

        # Strato B: bitcrush ──────────────────────────────────────────────────
        # Riduce i "livelli" disponibili → distorsione digitale/lo-fi
        if grit > 0:
            steps = max(2, int(2 + (1.0 - grit) * 16))
            grain = np.round(grain * steps) / steps

        # Inviluppo hanning: evita click tra grani sovrapposti
        env = np.hanning(g_len)
        glitch_layer[:, i : i + g_len] += grain * env * v_mix * (power + 0.1)

    # ── Strato C: drone 55 Hz ─────────────────────────────────────────────────
    t_axis = np.linspace(0, duration, N)
    drone = np.sin(2 * np.pi * 55 * t_axis) * params["drone_vol"] * (0.2 + e_map)
    drone_stereo = np.tile(drone, (2, 1))

    # ── Mix finale ────────────────────────────────────────────────────────────
    mix = (src * params["v_orig_vol"]) + (glitch_layer * 0.7) + drone_stereo
    return np.clip(mix, -1.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. EXPORT – scrive il WAV e lo fonde con il video originale
# ──────────────────────────────────────────────────────────────────────────────

def render(video_path: str, output_path: str, params: dict, sr: int = 44100):
    print(f"[1/3] Analisi video: {video_path}")
    energy = analyze_video(video_path)
    print(f"      Frames analizzati: {len(energy)}")

    clip = VideoFileClip(video_path)
    duration = clip.duration
    print(f"      Durata: {duration:.2f}s")

    print("[2/3] Sintesi audio glitch...")
    audio = generate_glitch_audio(video_path, energy, duration, params, sr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    sf.write(wav_path, audio.T, sr)   # soundfile vuole (N, 2)

    print(f"[3/3] Compositing → {output_path}")
    final = clip.set_audio(AudioFileClip(wav_path))
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)

    clip.close()
    os.unlink(wav_path)
    print("✅ Fatto!")


# ──────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "v_orig_vol": 0.3,   # volume audio originale
    "v_mix":      1.5,   # intensità strato granulare
    "grit":       0.7,   # bitcrush (0=pulito, 1=massimo)
    "g_size":     0.1,   # durata max grano (s)
    "intensity":  1.0,   # sensibilità all'energia video
    "drone_vol":  0.15,  # volume drone 55 Hz
    "seed":       42,    # riproducibilità
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BeatGlitch – audio-reactive glitching")
    ap.add_argument("--input",      required=True,  help="Video sorgente (.mp4 / .mov)")
    ap.add_argument("--output",     required=True,  help="Video output (.mp4)")
    ap.add_argument("--v_orig_vol", type=float, default=DEFAULT_PARAMS["v_orig_vol"])
    ap.add_argument("--v_mix",      type=float, default=DEFAULT_PARAMS["v_mix"])
    ap.add_argument("--grit",       type=float, default=DEFAULT_PARAMS["grit"])
    ap.add_argument("--g_size",     type=float, default=DEFAULT_PARAMS["g_size"])
    ap.add_argument("--intensity",  type=float, default=DEFAULT_PARAMS["intensity"])
    ap.add_argument("--drone_vol",  type=float, default=DEFAULT_PARAMS["drone_vol"])
    ap.add_argument("--seed",       type=int,   default=DEFAULT_PARAMS["seed"])
    args = ap.parse_args()

    params = {k: getattr(args, k) for k in DEFAULT_PARAMS}
    render(args.input, args.output, params)
