import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# ─────────────────────────────────────────────────────────────────
# 1. ANALISI VIDEO (Luce, Colore, Movimento, Variazione)
# ─────────────────────────────────────────────────────────────────

def analyze_video_v5(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sig = {"lum": [], "mot": [], "hue": [], "var": []}
    prev_gray = None
    prev_mot = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        img = cv2.resize(frame, (120, 90))
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
            sig["mot"].append(0.0)
            sig["var"].append(0.0)
        prev_gray = gray
    cap.release()

    for k in sig:
        arr = np.array(sig[k])
        sig[k] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return sig, fps

# ─────────────────────────────────────────────────────────────────
# 2. MOTORE AUDIO (Effetto Puntina Bloccatta / Stutter)
# ─────────────────────────────────────────────────────────────────

def generate_loop_glitch(video_path, sig, duration, p, sr=44100):
    np.random.seed(int(p["seed"]))
    N = int(duration * sr)
    t_ax = np.linspace(0, duration, N)
    
    # Mapping segnali
    mot = np.interp(t_ax, np.linspace(0, duration, len(sig["mot"])), sig["mot"])
    var = np.interp(t_ax, np.linspace(0, duration, len(sig["var"])), sig["var"])
    lum = np.interp(t_ax, np.linspace(0, duration, len(sig["lum"])), sig["lum"])

    # Caricamento Sorgente
    try:
        y_src, _ = librosa.load(video_path, sr=sr, mono=True)
        y_src = librosa.util.fix_length(y_src, size=N)
    except:
        y_src = np.random.uniform(-0.1, 0.1, N) # Fallback rumore se muto

    audio_glitch = np.zeros(N)
    
    # Parametri Stutter (il "salto della puntina")
    stutter_size = int(sr * (p["stutter_ms"] / 1000.0))
    stutter_reps = int(p["stutter_reps"])
    
    i = 0
    while i < N - stutter_size * stutter_reps:
        # Se c'è un picco di variazione (cambio frame netto) attiviamo il LOOP
        trigger = var[i] * p["intensity"]
        
        if trigger > 0.6: # SOGLIA PER IL LOOP
            # Prendiamo il "bit" di audio corrente
            chunk = y_src[i : i + stutter_size].copy()
            
            # Applichiamo GRIT (Bitcrush)
            if p["grit"] > 0:
                steps = max(2, int(2 + (1 - p["grit"]) * 12))
                chunk = np.round(chunk * steps) / steps
            
            # Ripetiamo il bit come una puntina bloccata
            for r in range(stutter_reps):
                start = i + (r * stutter_size)
                env = np.hanning(stutter_size)
                audio_glitch[start : start + stutter_size] += chunk * env * p["v_mix"]
            
            i += stutter_size * stutter_reps # Saltiamo avanti dopo il loop
        else:
            # Micro-glitch normale se non c'è il loop
            if mot[i] > 0.1:
                g_len = int(sr * 0.01)
                audio_glitch[i:i+g_len] += y_src[i:i+g_len] * p["v_mix"] * mot[i]
            i += int(sr * 0.005) # Avanzamento standard 5ms

    # Drone basato sulla luce
    drone = np.sin(2 * np.pi * 55 * t_ax) * p["drone_vol"] * lum
    
    # MIX FINALE SIMBIOTICO
    # Volume Originale + Glitch + Drone
    final = (y_src * p["v_orig_vol"]) + (audio_glitch * 0.7) + (drone * 0.3)
    return np.tile(np.clip(final, -0.9, 0.9), (2, 1))

# ─────────────────────────────────────────────────────────────────
# 3. INTERFACCIA STREAMLIT
# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch: Locked Groove", layout="wide")
st.title("🔊 BeatGlitch: Locked Groove & Nervous Sync")

with st.sidebar:
    st.header("📂 Sorgente")
    v_file = st.file_uploader("Video File", type=["mp4", "mov"])
    st.markdown("---")
    up_json = st.file_uploader("Carica Preset JSON", type="json")
    ld = json.load(up_json) if up_json else {}

st.subheader("🎛️ Mixer & Effetto Puntina")
c1, c2, c3 = st.columns(3)

with c1:
    v_orig_vol = st.slider("Volume Originale (Video)", 0.0, 1.0, ld.get("v_orig_vol", 0.3), key="v_orig_vol")
    v_mix = st.slider("Potenza Glitch/Loop", 0.0, 5.0, ld.get("v_mix", 2.0), key="v_mix")
    intensity = st.slider("Sensibilità ai Frame", 0.1, 3.0, ld.get("intensity", 1.5), key="intensity")

with c2:
    stutter_ms = st.slider("Durata Bit Loop (ms)", 10, 250, ld.get("stutter_ms", 60), key="stutter_ms")
    stutter_reps = st.slider("Ripetizioni Loop", 1, 20, ld.get("stutter_reps", 8), key="stutter_reps")
    grit = st.slider("Grit (Bitcrush)", 0.0, 1.0, ld.get("grit", 0.7), key="grit")

with c3:
    drone_vol = st.slider("Volume Drone (Luce)", 0.0, 1.0, ld.get("drone_vol", 0.2), key="drone_vol")
    seed = st.number_input("Seed", value=int(ld.get("seed", 42)), key="seed")
    if st.button("📥 Salva Preset"):
        p_now = {"v_orig_vol": v_orig_vol, "v_mix": v_mix, "intensity": intensity, "stutter_ms": stutter_ms, "stutter_reps": stutter_reps, "grit": grit, "drone_vol": drone_vol, "seed": seed}
        st.download_button("Scarica JSON", json.dumps(p_now), "preset.json")

if v_file:
    t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t_v.write(v_file.read())
    
    if st.button("🚀 GENERA VIDEO-REATTIVO", use_container_width=True):
        with st.status("Analisi pixel e generazione loop...") as status:
            sig, fps = analyze_video_v5(t_v.name)
            clip = VideoFileClip(t_v.name)
            
            p = {"v_orig_vol": v_orig_vol, "v_mix": v_mix, "intensity": intensity, "stutter_ms": stutter_ms, "stutter_reps": stutter_reps, "grit": grit, "drone_vol": drone_vol, "seed": seed, "g_size": 0.1}
            
            audio = generate_loop_glitch(t_v.name, sig, clip.duration, p)
            
            t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(t_a.name, audio.T, 44100)
            
            out_p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            clip.set_audio(AudioFileClip(t_a.name)).write_videofile(out_p, codec="libx264", audio_codec="aac", logger=None)
            
            st.video(out_p)
            status.update(label="✅ Sincronizzazione Completata!", state="complete")
