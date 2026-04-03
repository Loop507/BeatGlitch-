import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# ─────────────────────────────────────────────────────────────────
# 1. ANALISI DEI FLUSSI (ESTRAZIONE SENSORI)
# ─────────────────────────────────────────────────────────────────

def analyze_video_v7(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sig = {"lum": [], "mot": [], "hue": [], "var": []}
    prev_gray = None
    prev_mot = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
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

    for k in sig:
        arr = np.array(sig[k])
        # Normalizzazione dinamica per non avere mai "silenzio" se c'è immagine
        sig[k] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return sig, fps

# ─────────────────────────────────────────────────────────────────
# 2. MOTORE A GRANULAZIONE SINCRONA (L'ANIMA DEL SUONO)
# ─────────────────────────────────────────────────────────────────

def generate_perfect_sync(video_path, sig, duration, p, sr=44100):
    np.random.seed(int(p["seed"]))
    N = int(duration * sr)
    t_ax = np.linspace(0, duration, N)
    
    # Interpolazione dei segnali video sull'asse temporale audio
    mot = np.interp(t_ax, np.linspace(0, duration, len(sig["mot"])), sig["mot"])
    var = np.interp(t_ax, np.linspace(0, duration, len(sig["var"])), sig["var"])
    lum = np.interp(t_ax, np.linspace(0, duration, len(sig["lum"])), sig["lum"])
    hue = np.interp(t_ax, np.linspace(0, duration, len(sig["hue"])), sig["hue"])

    # Caricamento Audio Originale (Sorgente dei grani)
    try:
        y_src, _ = librosa.load(video_path, sr=sr, mono=True)
        y_src = librosa.util.fix_length(y_src, size=N)
    except:
        y_src = np.random.normal(0, 0.2, N) # Se muto, usa rumore bianco come materia prima

    out_audio = np.zeros(N)
    
    # Parametri Stutter
    stutter_len = int(sr * (p["stutter_ms"] / 1000.0))
    stutter_reps = int(p["stutter_reps"])
    
    idx = 0
    while idx < N - max(stutter_len * stutter_reps, 1000):
        # 1. CONTROLLO LOOP (PUNTINA BLOCCATA)
        # Si attiva se la variazione (var) o il flash (lum) superano la soglia
        if var[idx] * p["intensity"] > 0.75:
            # Catturiamo il frammento ESATTO di audio di quel momento
            fragment = y_src[idx : idx + stutter_len].copy()
            
            # Applichiamo Bitcrush al frammento
            steps = max(2, int(2 + (1 - p["grit"]) * 15))
            fragment = np.round(fragment * steps) / steps
            
            for r in range(stutter_reps):
                pos = idx + (r * stutter_len)
                if pos + stutter_len < N:
                    # Inviluppo per evitare "clic" fastidiosi tra i loop
                    env = np.hanning(stutter_len)
                    out_audio[pos : pos + stutter_len] += fragment * env * p["v_mix"]
            idx += stutter_len * stutter_reps
            
        # 2. CONTROLLO MICRO-GLITCH (SINCRONIA CONTINUA)
        else:
            # Più l'immagine si muove (mot), più grani generiamo
            if mot[idx] > 0.05:
                # Dimensione grano reattiva
                g_size = int(sr * np.random.uniform(0.005, 0.04))
                # Estrazione grano dall'audio originale in quel preciso istante
                grain = y_src[idx : idx + g_size].copy()
                
                # Modulazione timbrica basata sul colore (Hue)
                res = int(2 + (1-p["grit"]) * 20 * (1-hue[idx]))
                grain = np.round(grain * res) / res
                
                out_audio[idx : idx + g_size] += grain * np.hanning(g_size) * p["v_mix"] * (mot[idx] * p["intensity"])
            
            # Avanzamento veloce (micro-step)
            idx += int(sr * 0.005) # 5ms

    # Drone armonico (la "colla" sonora) che segue la luminosità
    drone = np.sin(2 * np.pi * (40 + hue * 100) * t_ax) * p["drone_vol"] * lum
    
    # MIX FINALE: Audio Originale + Processato
    final = (y_src * p["v_orig_vol"]) + (out_audio * 0.8) + (drone * 0.2)
    return np.tile(np.clip(final, -1.0, 1.0), (2, 1))

# ─────────────────────────────────────────────────────────────────
# 3. INTERFACCIA STREAMLIT PRO
# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch V7", layout="wide")
st.title("🧪 BeatGlitch V7: Perfect Granular Sync")

with st.sidebar:
    v_file = st.file_uploader("Video Sorgente", type=["mp4", "mov"])
    st.markdown("---")
    up_json = st.file_uploader("Importa Preset JSON", type="json")
    ld = json.load(up_json) if up_json else {}

st.subheader("🎛️ Pannello di Controllo Simbiotico")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Mixer Sorgente**")
    v_orig_vol = st.slider("Volume Originale (Base)", 0.0, 1.0, ld.get("v_orig_vol", 0.3), key="v_orig_vol")
    v_mix = st.slider("Potenza Glitch/Sync", 0.0, 5.0, ld.get("v_mix", 2.0), key="v_mix")
    intensity = st.slider("Sensibilità Reattiva", 0.1, 4.0, ld.get("intensity", 1.8), key="intensity")

with col2:
    st.write("**Locked Groove (Loop)**")
    stutter_ms = st.slider("Durata Bit (ms)", 5, 250, ld.get("stutter_ms", 50), key="stutter_ms")
    stutter_reps = st.slider("N. Ripetizioni", 1, 32, ld.get("stutter_reps", 12), key="stutter_reps")

with col3:
    st.write("**Timbro**")
    grit = st.slider("Grit (Saturazione)", 0.0, 1.0, ld.get("grit", 0.8), key="grit")
    drone_vol = st.slider("Volume Drone (Luce)", 0.0, 1.0, ld.get("drone_vol", 0.1), key="drone_vol")
    seed = st.number_input("Seed", value=int(ld.get("seed", 42)), key="seed")

if v_file:
    t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t_v.write(v_file.read())
    
    if st.button("🚀 GENERA SINCRONIZZAZIONE PERFETTA", use_container_width=True):
        with st.status("Analisi frame-by-frame e sintesi granulare...") as status:
            sig, fps = analyze_video_v7(t_v.name)
            clip = VideoFileClip(t_v.name)
            p = {"v_orig_vol":v_orig_vol, "v_mix":v_mix, "intensity":intensity, "stutter_ms":stutter_ms, "stutter_reps":stutter_reps, "grit":grit, "drone_vol":drone_vol, "seed":seed}
            
            audio = generate_perfect_sync(t_v.name, sig, clip.duration, p)
            t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(t_a.name, audio.T, 44100)
            
            out_p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            clip.set_audio(AudioFileClip(t_a.name)).write_videofile(out_p, codec="libx264", audio_codec="aac", logger=None)
            
            st.video(out_p)
            status.update(label="✅ Sincronizzazione Perfetta!", state="complete")
