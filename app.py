import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# ─────────────────────────────────────────────────────────────────
# 1. ANALISI DEI SENSORI (Luce, Colore, Movimento, Variazione)
# ─────────────────────────────────────────────────────────────────

def analyze_video_v9(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sig = {"lum": [], "mot": [], "hue": [], "var": []}
    prev_gray = None
    prev_mot = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Resize per velocità e densità analisi
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

    # Normalizzazione per reattività massima
    for k in sig:
        arr = np.array(sig[k])
        sig[k] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return sig, fps

# ─────────────────────────────────────────────────────────────────
# 2. MOTORE IBRIDO (Sorgente Reale + Sintesi d'Emergenza)
# ─────────────────────────────────────────────────────────────────

def generate_v9_engine(video_path, sig, duration, p, sr=44100):
    np.random.seed(int(p["seed"]))
    N = int(duration * sr)
    t_ax = np.linspace(0, duration, N)
    
    # Mapping segnali video su asse audio
    mot = np.interp(t_ax, np.linspace(0, duration, len(sig["mot"])), sig["mot"])
    var = np.interp(t_ax, np.linspace(0, duration, len(sig["var"])), sig["var"])
    lum = np.interp(t_ax, np.linspace(0, duration, len(sig["lum"])), sig["lum"])
    hue = np.interp(t_ax, np.linspace(0, duration, len(sig["hue"])), sig["hue"])

    # --- TENTATIVO CARICAMENTO AUDIO ---
    try:
        y_src, _ = librosa.load(video_path, sr=sr, mono=True)
        if np.max(np.abs(y_src)) < 1e-3: raise ValueError("Muto")
        y_src = librosa.util.fix_length(y_src, size=N)
    except:
        # Se il video è muto, generiamo una 'polvere sonora' (Hiss)
        # che respira con la luminosità del video
        y_src = np.random.uniform(-0.4, 0.4, N) * lum

    out_glitch = np.zeros(N)
    stutter_len = int(sr * (p["stutter_ms"] / 1000.0))
    stutter_reps = int(p["stutter_reps"])
    
    idx = 0
    while idx < N - max(stutter_len * stutter_reps, 1000):
        energy = mot[idx] * p["intensity"]
        
        # 1. TRIGGER LOOP (Puntina bloccata) su variazioni brusche
        if var[idx] * p["intensity"] > 0.8:
            fragment = y_src[idx : idx + stutter_len].copy()
            res = max(2, int(2 + (1-p["grit"])*15))
            fragment = np.round(fragment * res) / res
            
            for r in range(stutter_reps):
                pos = idx + (r * stutter_len)
                if pos + stutter_len < N:
                    out_glitch[pos : pos + stutter_len] += fragment * np.hanning(stutter_len) * p["v_mix"]
            idx += stutter_len * stutter_reps
            
        # 2. TRIGGER TEMPESTA (Granulazione Sincrona)
        elif energy > 0.1:
            g_len = int(sr * np.random.uniform(0.005, 0.04))
            grain = y_src[idx : idx + g_len].copy()
            # Il bitcrush segue il colore: più hue = suono più sporco
            grit_res = max(2, int(2 + (1-p["grit"]) * 20 * (1-hue[idx])))
            grain = np.round(grain * grit_res) / grit_res
            
            out_glitch[idx : idx + g_len] += grain * np.hanning(g_len) * p["v_mix"] * energy
            idx += int(sr * 0.003) # Passo denso per tempesta continua
        else:
            idx += int(sr * 0.005)

    # Drone armonico (Frequenza basata sul colore, Volume sulla luce)
    f_drone = 40 + (hue * 120)
    drone = np.sin(2 * np.pi * f_drone * t_ax) * p["drone_vol"] * lum
    
    # MIX FINALE
    final = (y_src * p["v_orig_vol"]) + (out_glitch * 0.8) + (drone * 0.2)
    return np.tile(np.clip(final, -1.0, 1.0), (2, 1))

# ─────────────────────────────────────────────────────────────────
# 3. INTERFACCIA STREAMLIT
# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch V9", layout="wide")
st.title("⚡ BeatGlitch V9: Quantum Symbiosis")

with st.sidebar:
    st.header("📂 Risorse")
    v_file = st.file_uploader("Carica Video", type=["mp4", "mov"])
    st.markdown("---")
    up_json = st.file_uploader("Carica Preset JSON", type="json")
    ld = json.load(up_json) if up_json else {}

st.subheader("🎛️ Mixer della Tempesta Reattiva")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Mixer Sorgente**")
    v_orig_vol = st.slider("Volume Originale (Base)", 0.0, 1.0, ld.get("v_orig_vol", 0.3), key="v_orig_vol")
    v_mix = st.slider("Potenza Tempesta (Glitch)", 0.0, 5.0, ld.get("v_mix", 2.5), key="v_mix")
    intensity = st.slider("Sensibilità Reattiva", 0.1, 4.0, ld.get("intensity", 1.8), key="intensity")

with col2:
    st.write("**Effetto Puntina (Loop)**")
    stutter_ms = st.slider("Durata Loop (ms)", 5, 200, ld.get("stutter_ms", 45), key="stutter_ms")
    stutter_reps = st.slider("N. Ripetizioni", 1, 40, ld.get("stutter_reps", 12), key="stutter_reps")

with col3:
    st.write("**Timbro**")
    grit = st.slider("Grit (Bitcrush)", 0.0, 1.0, ld.get("grit", 0.8), key="grit")
    drone_vol = st.slider("Volume Drone (Luce)", 0.0, 1.0, ld.get("drone_vol", 0.15), key="drone_vol")
    seed = st.number_input("Seed", value=int(ld.get("seed", 42)), key="seed")

if v_file:
    t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t_v.write(v_file.read())
    
    if st.button("🚀 GENERA SINCRONIZZAZIONE TOTALE", use_container_width=True):
        with st.status("Analisi frame-by-frame e sintesi ibrida...") as status:
            sig, fps = analyze_video_v9(t_v.name)
            clip = VideoFileClip(t_v.name)
            p = {"v_orig_vol":v_orig_vol, "v_mix":v_mix, "intensity":intensity, "stutter_ms":stutter_ms, "stutter_reps":stutter_reps, "grit":grit, "drone_vol":drone_vol, "seed":seed}
            
            audio = generate_v9_engine(t_v.name, sig, clip.duration, p)
            t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(t_a.name, audio.T, 44100)
            
            out_p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            clip.set_audio(AudioFileClip(t_a.name)).write_videofile(out_p, codec="libx264", audio_codec="aac", logger=None)
            
            st.video(out_p)
            status.update(label="✅ Sincronizzazione Completata!", state="complete")
            with open(out_p, "rb") as f:
                st.download_button("💾 Scarica Video Finale", f, "beatglitch_v9.mp4")
