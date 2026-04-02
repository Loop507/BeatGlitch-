"""
BeatGlitch – Audio-Reactive Glitching
Interfaccia Streamlit. Avvia con: streamlit run app.py

Dipendenze (requirements.txt):
    opencv-python-headless
    librosa
    soundfile
    moviepy
    numpy
"""

import time
import json
import os
import tempfile

import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip


# ──────────────────────────────────────────────────────────────────────────────
# 1. ANALISI VIDEO – energia frame-per-frame
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video(video_path: str, res=(160, 120)) -> np.ndarray:
    """
    Energia per frame combinando 3 segnali:
      A) Movimento (diff luminanza)        – come prima
      B) Shift di colore (diff HSV hue)    – cambi di tinta, luci colorate
      C) Taglio di scena (spike istogramma) – transizioni brusche
    I tre canali vengono normalizzati e mixati con pesi diversi.
    """
    cap = cv2.VideoCapture(video_path)
    motion_s, color_s, hist_s = [], [], []
    prev_gray = prev_hue = prev_hist = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, res)

        # A) Movimento (luminanza)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_s.append(float(np.mean(diff) + np.std(diff)))
        else:
            motion_s.append(0.0)
        prev_gray = gray

        # B) Shift di colore (canale Hue in HSV)
        hsv  = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue  = hsv[:, :, 0].astype(np.float32)
        if prev_hue is not None:
            # differenza circolare su [0,180]
            d = np.abs(hue - prev_hue)
            d = np.minimum(d, 180 - d)
            color_s.append(float(np.mean(d) + np.std(d)))
        else:
            color_s.append(0.0)
        prev_hue = hue

        # C) Taglio di scena (distanza istogramma BGR)
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        if prev_hist is not None:
            hist_s.append(float(np.sum(np.abs(hist - prev_hist))))
        else:
            hist_s.append(0.0)
        prev_hist = hist

    cap.release()

    def norm(x):
        a = np.array(x, dtype=np.float32)
        mx = a.max()
        return a / mx if mx > 0 else np.full_like(a, 0.1)

    motion = norm(motion_s)
    color  = norm(color_s)
    scene  = norm(hist_s)

    # Pesi: movimento 40%, colore 35%, taglio scena 25%
    combined = 0.40 * motion + 0.35 * color + 0.25 * scene

    # Normalizza finale + curva che enfatizza i picchi
    mx = combined.max()
    if mx > 0:
        combined = combined / mx
    combined = np.power(combined, 1.3)   # meno aggressivo di 1.5 perché già ricco
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# 2. SINTESI AUDIO GLITCH (vettorizzata)
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_audio(video_path, energy, duration, params, sr=44100):
    np.random.seed(int(params["seed"]))
    N = int(duration * sr)

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

    e_map = np.interp(
        np.linspace(0, 1, N),
        np.linspace(0, 1, len(energy)),
        energy
    ).astype(np.float32)

    intensity  = params["intensity"]
    v_mix      = params["v_mix"]
    grit       = params["grit"]
    G_LEN      = int(sr * params["g_size"])
    N_GRAINS   = int(duration / 0.005)

    starts = np.random.randint(0, max(1, N - G_LEN), size=N_GRAINS)
    powers = e_map[starts] * intensity
    keep   = (powers > 0.02) | (np.random.random(N_GRAINS) < 0.05)
    starts, powers = starts[keep], powers[keep]

    idx_matrix = (starts[:, None] + np.arange(G_LEN)).clip(0, N - 1)
    grains_l = src[0][idx_matrix]
    grains_r = src[1][idx_matrix]

    if grit > 0:
        steps = max(2, int(2 + (1.0 - grit) * 16))
        grains_l = np.round(grains_l * steps) / steps
        grains_r = np.round(grains_r * steps) / steps

    env   = np.hanning(G_LEN).astype(np.float32)
    scale = (powers + 0.1)[:, None] * env * v_mix
    grains_l *= scale
    grains_r *= scale

    glitch_l = np.zeros(N, dtype=np.float32)
    glitch_r = np.zeros(N, dtype=np.float32)
    flat_idx = idx_matrix.ravel()
    np.add.at(glitch_l, flat_idx, grains_l.ravel())
    np.add.at(glitch_r, flat_idx, grains_r.ravel())
    glitch_layer = np.stack([glitch_l, glitch_r])

    t_axis = np.linspace(0, duration, N, dtype=np.float32)
    drone  = np.sin(2 * np.pi * 55 * t_axis) * params["drone_vol"] * (0.2 + e_map)
    drone_stereo = np.tile(drone, (2, 1))

    mix = (src * params["v_orig_vol"]) + (glitch_layer * 0.7) + drone_stereo
    return np.clip(mix, -1.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. INTERFACCIA STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch Studio", layout="wide")
st.title("🎬 BeatGlitch – Audio-Reactive Glitching")

# Sidebar
with st.sidebar:
    st.header("📂 Risorse")
    v_file  = st.file_uploader("Carica Video", type=["mp4", "mov"])
    st.markdown("---")
    st.subheader("💾 Preset JSON")
    up_json = st.file_uploader("Carica Preset", type=["json"])

    if st.button("📥 Prepara Download Preset"):
        curr = {k: st.session_state.get(k, v) for k, v in {
            "v_orig_vol": 0.3, "v_mix": 1.5, "grit": 0.7,
            "g_size": 0.1, "intensity": 1.0, "drone_vol": 0.15, "seed": 42
        }.items()}
        st.download_button("Scarica Ora", json.dumps(curr, indent=2), "preset.json")

# Carica valori da JSON se fornito
ld = {}
if up_json:
    try:
        ld = json.load(up_json)
    except Exception:
        st.sidebar.warning("Preset JSON non valido.")

# Controlli
st.subheader("🎛️ Controlli Audio-Visivi")
c1, c2, c3 = st.columns(3)
with c1:
    v_orig_vol = st.slider("Volume Originale Video",   0.0, 1.0, float(ld.get("v_orig_vol", 0.3)), key="v_orig_vol")
    v_mix      = st.slider("Potenza Glitch Generato",  0.0, 5.0, float(ld.get("v_mix",      1.5)), key="v_mix")
with c2:
    grit   = st.slider("Grit (Bitcrush)",              0.0, 1.0, float(ld.get("grit",   0.7)), key="grit")
    g_size = st.slider("Durata Micro-Glitch (s)",      0.01, 0.5, float(ld.get("g_size", 0.1)), key="g_size")
with c3:
    intensity  = st.slider("Sensibilità ai Pixel",     0.0, 2.0, float(ld.get("intensity",  1.0)), key="intensity")
    drone_vol  = st.slider("Volume Drone (55 Hz)",     0.0, 1.0, float(ld.get("drone_vol", 0.15)), key="drone_vol")
    seed       = st.number_input("Seed",               value=int(ld.get("seed", 42)),               key="seed")

# Generazione
if v_file:
    # Salva video in un file temporaneo persistente durante la sessione
    if "tmp_video_path" not in st.session_state:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        t_v.flush()
        st.session_state["tmp_video_path"] = t_v.name
    video_path = st.session_state["tmp_video_path"]

    if st.button("🚀 GENERA GLITCH ART", use_container_width=True):
        params = {
            "v_orig_vol": v_orig_vol, "v_mix": v_mix, "grit": grit,
            "g_size": g_size, "intensity": intensity,
            "drone_vol": drone_vol, "seed": seed,
        }

        try:
            prog = st.progress(0, text="[1/4] Analisi video...")
            t0 = time.time()
            energy = analyze_video(video_path)
            st.caption(f"✓ Frames: {len(energy)}  ({time.time()-t0:.1f}s)")
            prog.progress(25, text="[2/4] Apertura clip...")

            clip     = VideoFileClip(video_path)
            duration = clip.duration
            st.caption(f"✓ Durata: {duration:.2f}s")
            prog.progress(40, text="[3/4] Sintesi audio glitch...")

            t0    = time.time()
            audio = generate_glitch_audio(video_path, energy, duration, params)
            st.caption(f"✓ Audio sintetizzato  ({time.time()-t0:.1f}s)")
            prog.progress(70, text="[4/4] Export video finale...")

            wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(wav_tmp.name, audio.T, 44100)

            out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            t0 = time.time()
            final_clip = clip.set_audio(AudioFileClip(wav_tmp.name))
            final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
            st.caption(f"✓ Video scritto  ({time.time()-t0:.1f}s)")

            clip.close()
            os.unlink(wav_tmp.name)
            prog.progress(100, text="✅ Completato!")

            st.success("Sincronizzazione completata!")
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("💾 Scarica Video Finale", f, "glitch_video.mp4", mime="video/mp4")

        except Exception as e:
            st.error(f"Errore durante la generazione: {e}")
            st.exception(e)

else:
    st.info("⬆️ Carica un video nella sidebar per iniziare.")
