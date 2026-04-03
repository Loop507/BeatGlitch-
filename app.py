import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# ─────────────────────────────────────────────────────────────────
# 1. ANALISI SENSORI (Luce, Colore, Movimento, Variazione)
# ─────────────────────────────────────────────────────────────────

def analyze_video_v10(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sig = {"lum": [], "mot": [], "hue": [], "var": []}
    prev_gray = None
    prev_mot = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        # Analisi densa per catturare ogni micro-strappo
        img = cv2.resize(frame, (100, 75))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        sig["lum"].append(np.mean(gray) / 255.0)
        sig["hue"].append(np.mean(hsv[:,:,0]) / 180.0)
        
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            m = np.mean(diff) / 255.0
            sig["mot"].append(m)
            sig["var"].append(abs(m - prev_mot))
            prev_mot = m
        else:
            sig["mot"].append(0.0); sig["var"].append(0.0)
        prev_gray = gray
    cap.release()

    # Normalizzazione per reattività estrema
    for k in sig:
        arr = np.array(sig[k])
        sig[k] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return sig, fps

# ─────────────────────────────────────────────────────────────────
# 2. MOTORE DI DE-COSTRUZIONE (Quantum Shredder)
# ─────────────────────────────────────────────────────────────────

def generate_v10_engine(video_path, audio_ext_path, sig, duration, p, sr=44100):
    np.random.seed(int(p["seed"]))
    N = int(duration * sr)
    t_ax = np.linspace(0, duration, N)
    
    # Mapping segnali video su asse temporale audio
    mot = np.interp(t_ax, np.linspace(0, duration, len(sig["mot"])), sig["mot"])
    var = np.interp(t_ax, np.linspace(0, duration, len(sig["var"])), sig["var"])
    lum = np.interp(t_ax, np.linspace(0, duration, len(sig["lum"])), sig["lum"])
    hue = np.interp(t_ax, np.linspace(0, duration, len(sig["hue"])), sig["hue"])

    # Caricamento Sorgente (Esterna o Interna)
    try:
        path_to_load = audio_ext_path if audio_ext_path else video_path
        y_src, _ = librosa.load(path_to_load, sr=sr, mono=True)
        
        # Gestione Lunghezza (Auto-Trim o Auto-Loop)
        if len(y_src) > N:
            y_src = y_src[:N] # Taglia se troppo lungo
        else:
            y_src = np.tile(y_src, int(np.ceil(N/len(y_src))))[:N] # Loop se troppo corto
            
        if np.max(np.abs(y_src)) < 1e-3: raise ValueError("Silenzio")
    except:
        # Sintesi di emergenza (Hiss reattivo)
        y_src = np.random.uniform(-0.3, 0.3, N) * lum

    out_glitch = np.zeros(N)
    stutter_len = int(sr * (p["stutter_ms"] / 1000.0))
    stutter_reps = int(p["stutter_reps"])
    
    idx = 0
    while idx < N - max(stutter_len * stutter_reps, 2000):
        energy = mot[idx] * p["intensity"]
        
        # --- TRIGGER 1: LOCKED GROOVE (Il disco salta) ---
        if var[idx] * p["intensity"] > 0.8:
            fragment = y_src[idx : idx + stutter_len].copy()
            # Bitcrush basato sul parametro Grit
            res = max(2, int(2 + (1-p["grit"])*16))
            fragment = np.round(fragment * res) / res
            
            for r in range(stutter_reps):
                pos = idx + (r * stutter_len)
                if pos + stutter_len < N:
                    out_glitch[pos : pos + stutter_len] += fragment * np.hanning(stutter_len) * p["v_mix"]
            idx += stutter_len * stutter_reps
            
        # --- TRIGGER 2: TEMPESTA GRANULARE (Sincronia millimetrica) ---
        elif energy > 0.1:
            g_len = int(sr * np.random.uniform(0.005, 0.04))
            grain = y_src[idx : idx + g_len].copy()
            # Il colore influenza la distorsione
            grit_res = max(2, int(2 + (1-p["grit"]) * 20 * (1-hue[idx])))
            grain = np.round(grain * grit_res) / grit_res
            
            out_glitch[idx : idx + g_len] += grain * np.hanning(g_len) * p["v_mix"] * energy
            idx += int(sr * 0.002) # Passo ultra-denso
        else:
            idx += int(sr * 0.005)

    # Drone di fondo (armonici basati sul colore)
    f_drone = 40 + (hue * 120)
    drone = np.sin(2 * np.pi * f_drone * t_ax) * p["drone_vol"] * lum
    
    # Mix Finale: Canale Sinistro e Destro (lieve panning reattivo)
    final_mono = (y_src * p["v_orig_vol"]) + (out_glitch * 0.8) + (drone * 0.2)
    final_stereo = np.tile(np.clip(final_mono, -1.0, 1.0), (2, 1))
    
    return final_stereo

# ─────────────────────────────────────────────────────────────────
# 3. INTERFACCIA STREAMLIT V10
# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch V10 Pro", layout="wide")
st.title("🌪️ BeatGlitch V10: Quantum Shredder")

with st.sidebar:
    st.header("📂 Sorgenti")
    v_file = st.file_uploader("1. Video (Immagini)", type=["mp4", "mov"])
    a_file = st.file_uploader("2. Audio Esterno (Opzionale)", type=["mp3", "wav", "m4a"])
    st.markdown("---")
    up_json = st.file_uploader("Carica Preset JSON", type="json")
    ld = json.load(up_json) if up_json else {}

st.subheader("🎛️ Controllo della Distruzione")
col1, col2, col3 = st.columns(3)

with col1:
    v_orig_vol = st.slider("Volume Sorgente (Base)", 0.0, 1.0, ld.get("v_orig_vol", 0.2))
    v_mix = st.slider("Potenza Tempesta Glitch", 0.0, 5.0, ld.get("v_mix", 2.8))
    intensity = st.slider("Sensibilità ai Pixel", 0.1, 4.0, ld.get("intensity", 2.0))

with col2:
    stutter_ms = st.slider("Durata Salto Puntina (ms)", 5, 250, ld.get("stutter_ms", 40))
    stutter_reps = st.slider("N. Ripetizioni Loop", 1, 50, ld.get("stutter_reps", 16))

with col3:
    grit = st.slider("Grit (De-costruzione)", 0.0, 1.0, ld.get("grit", 0.85))
    drone_vol = st.slider("Volume Drone (Luce)", 0.0, 1.0, ld.get("drone_vol", 0.15))
    seed = st.number_input("Seed (Identità)", value=int(ld.get("seed", 42)))

if v_file:
    t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t_v.write(v_file.read())
    
    t_a_ext = None
    if a_file:
        suffix = f".{a_file.name.split('.')[-1]}"
        t_a_ext = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        t_a_ext.write(a_file.read())

    if st.button("🚀 AVVIA DE-COSTRUZIONE SIMBIOTICA", use_container_width=True):
        with st.status("Processando pixel e atomi sonori...") as status:
            sig, fps = analyze_video_v10(t_v.name)
            clip = VideoFileClip(t_v.name)
            p = {"v_orig_vol":v_orig_vol, "v_mix":v_mix, "intensity":intensity, 
                 "stutter_ms":stutter_ms, "stutter_reps":stutter_reps, "grit":grit, 
                 "drone_vol":drone_vol, "seed":seed}
            
            ext_path = t_a_ext.name if t_a_ext else None
            audio = generate_v10_engine(t_v.name, ext_path, sig, clip.duration, p)
            
            t_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(t_wav.name, audio.T, 44100)
            
            out_p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            clip.set_audio(AudioFileClip(t_wav.name)).write_videofile(out_p, codec="libx264", audio_codec="aac", logger=None)
            
            st.video(out_p)
            status.update(label="✅ Distruzione completata!", state="complete")
            with open(out_p, "rb") as f:
                st.download_button("💾 Scarica Risultato", f, "shredder_v10.mp4")
