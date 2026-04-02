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
    Sintesi vettorizzata – nessun loop Python, usa NumPy per tutto.
    Ritorna array stereo float32 shape (2, N).
    """
    np.random.seed(int(params["seed"]))
    N = int(duration * sr)

    # ── Caricamento audio sorgente ────────────────────────────────────────────
    try:
        src, _ = librosa.load(video_path, sr=sr, mono=False)
        if src.ndim == 1:
            src = np.tile(src, (2, 1))
        if src.shape[1] < N:
            src = np.pad(src, ((0, 0), (0, N - src.shape[1])))
        else:
            src = src[:, :N]
    except Exception:
        src = np.zeros((2, N), dtype=np.float32)

    # ── Mappa energia video → campioni audio ──────────────────────────────────
    e_map = np.interp(
        np.linspace(0, 1, N),
        np.linspace(0, 1, len(energy)),
        energy
    ).astype(np.float32)

    # ── Strato A: glitch granulare vettorizzato ───────────────────────────────
    # Invece di iterare ogni 5ms, scegliamo N_GRAINS posizioni di partenza
    # casuali e le sommiamo in un'unica operazione vettoriale.
    intensity  = params["intensity"]
    v_mix      = params["v_mix"]
    grit       = params["grit"]
    g_size_max = params["g_size"]
    G_LEN      = int(sr * g_size_max)          # lunghezza fissa per la vettorizzazione
    N_GRAINS   = int(duration / 0.005)         # ~1 grano ogni 5ms (come prima)

    # Posizioni di partenza dei grani (campioni)
    starts = np.random.randint(0, max(1, N - G_LEN), size=N_GRAINS)
    # Energia in quel punto → "power" del grano
    powers = e_map[starts] * intensity

    # Teniamo solo i grani sopra soglia o con probabilità 5%
    keep = (powers > 0.02) | (np.random.random(N_GRAINS) < 0.05)
    starts = starts[keep]
    powers = powers[keep]

    # Costruiamo la matrice dei grani: shape (n_kept, G_LEN)
    idx_matrix = (starts[:, None] + np.arange(G_LEN)).clip(0, N - 1)  # (K, G_LEN)
    grains_l = src[0][idx_matrix]   # canale L
    grains_r = src[1][idx_matrix]   # canale R

    # Bitcrush vettorizzato
    if grit > 0:
        steps = max(2, int(2 + (1.0 - grit) * 16))
        grains_l = np.round(grains_l * steps) / steps
        grains_r = np.round(grains_r * steps) / steps

    # Inviluppo Hanning (evita click)
    env = np.hanning(G_LEN).astype(np.float32)             # (G_LEN,)
    scale = (powers + 0.1)[:, None] * env * v_mix          # (K, G_LEN)
    grains_l *= scale
    grains_r *= scale

    # Accumulo: add.at è lento, usiamo np.zeros + bincount trick per L/R
    glitch_l = np.zeros(N, dtype=np.float32)
    glitch_r = np.zeros(N, dtype=np.float32)
    flat_idx = idx_matrix.ravel()                           # (K*G_LEN,)
    np.add.at(glitch_l, flat_idx, grains_l.ravel())
    np.add.at(glitch_r, flat_idx, grains_r.ravel())
    glitch_layer = np.stack([glitch_l, glitch_r])           # (2, N)

    # ── Strato C: drone 55 Hz modulato dall'energia ───────────────────────────
    t_axis = np.linspace(0, duration, N, dtype=np.float32)
    drone  = np.sin(2 * np.pi * 55 * t_axis) * params["drone_vol"] * (0.2 + e_map)
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
