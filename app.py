import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# ─────────────────────────────────────────────────────────────────
# 1. ANALISI AVANZATA (Derivata dal tuo VSG)
# ─────────────────────────────────────────────────────────────────

def analyze_video_pro(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    signals = {"lum": [], "det": [], "mot": [], "hue": [], "sat": []}
    prev_gray = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Ridimensionamento per analisi rapida
        img = cv2.resize(frame, (160, 120))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Estrazione parametri (Simbiosi totale)
        signals["lum"].append(np.mean(gray) / 255.0)      # Luce
        signals["det"].append(np.std(gray) / 255.0)       # Complessità visiva
        signals["hue"].append(np.mean(hsv[:,:,0]) / 180.0) # Tono colore
        signals["sat"].append(np.mean(hsv[:,:,1]) / 255.0) # Saturazione
        
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            signals["mot"].append(np.mean(diff) / 255.0)
        else:
            signals["mot"].append(0.0)
        prev_gray = gray
    cap.release()

    # Normalizzazione segnali
    for k in signals:
        arr = np.array(signals[k])
        mx = arr.max()
        signals[k] = arr / mx if mx > 0 else np.zeros_like(arr)
        
    return signals, fps

# ─────────────────────────────────────────────────────────────────
# 2. MOTORE DI SINTESI SIMBIOTICA (Glitch + FM Synthesis)
# ─────────────────────────────────────────────────────────────────

def generate_symbiotic_audio(video_path, signals, duration, p, sr=44100):
    np.random.seed(int(p["seed"]))
    N = int(duration * sr)
    t_ax = np.linspace(0, duration, N)
    
    # Mappatura dei segnali sui campioni audio
    def get_map(sig_name):
        return np.interp(t_ax, np.linspace(0, duration, len(signals[sig_name])), signals[sig_name])

    lum = get_map("lum")
    mot = get_map("mot")
    det = get_map("det")
    hue = get_map("hue")
    
    # --- CANALE 1: SINTESI DAL NULLA (Per video muti) ---
    # Usiamo la frequenza basata sul colore (HUE) e l'ampiezza sulla luminosità
    base_freq = 40 + (hue * 400) # Il colore decide la nota
    carrier = np.sin(2 * np.pi * base_freq * t_ax)
    # Modulazione FM "sporca" basata sul dettaglio visivo
    modulator = np.sin(2 * np.pi * (base_freq * 1.5) * t_ax) * det * 10
    synth_tone = np.sin(2 * np.pi * base_freq * t_ax + modulator)
    
    # --- CANALE 2: GRANULAZIONE (Se esiste audio) ---
    glitch_layer = np.zeros(N)
    try:
        y_src, _ = librosa.load(video_path, sr=sr, mono=True)
        y_src = librosa.util.fix_length(y_src, size=N)
        
        # Creiamo burst di glitch sui picchi di movimento (MOT)
        step = int(sr * 0.01) # 10ms
        for i in range(0, N - int(sr*p["g_size"]), step):
            if mot[i] > (0.1 / p["intensity"]):
                g_len = int(sr * np.random.uniform(0.005, p["g_size"]))
                chunk = y_src[i : i + g_len]
                # Bitcrush (Grit)
                if p["grit"] > 0:
                    s = max(2, int(2 + (1-p["grit"])*16))
                    chunk = np.round(chunk * s) / s
                glitch_layer[i : i + g_len] += chunk * np.hanning(g_len)
    except:
        glitch_layer = synth_tone * mot # Se muto, il synth segue il movimento
        
    # --- MIX FINALE ---
    # Il drone segue la luminosità, il glitch segue il movimento
    out = (synth_tone * p["drone_vol"] * lum) + (glitch_layer * p["v_mix"])
    
    # Limiter soft
    out = np.clip(out, -0.9, 0.9)
    return np.tile(out, (2, 1)) # Stereo

# ─────────────────────────────────────────────────────────────────
# 3. INTERFACCIA
# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch PRO", layout="wide")
st.title("🧩 BeatGlitch PRO: Symbiotic Engine")

with st.sidebar:
    v_file = st.file_uploader("Carica Video", type=["mp4", "mov"])
    st.markdown("---")
    # Caricamento Preset (Mantenuto dal tuo stile)
    up_json = st.file_uploader("Carica Preset JSON", type="json")
    ld = json.load(up_json) if up_json else {}

st.subheader("🎛️ Parametri di Simbiosi")
c1, c2, c3 = st.columns(3)
with c1:
    v_mix = st.slider("Potenza Glitch (Movimento)", 0.0, 5.0, ld.get("v_mix", 2.0), key="v_mix")
    intensity = st.slider("Sensibilità Trigger", 0.1, 2.0, ld.get("intensity", 1.0), key="intensity")
with c2:
    grit = st.slider("Grit (Bitcrush)", 0.0, 1.0, ld.get("grit", 0.7), key="grit")
    g_size = st.slider("Max Grain Size", 0.01, 0.4, ld.get("g_size", 0.1), key="g_size")
with c3:
    drone_vol = st.slider("Drone (Luminosità/Colore)", 0.0, 1.0, ld.get("drone_vol", 0.3), key="drone_vol")
    seed = st.number_input("Seed", value=int(ld.get("seed", 42)), key="seed")

if v_file:
    t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t_v.write(v_file.read())
    
    if st.button("🚀 GENERA SINCRONIZZAZIONE TOTALE", use_container_width=True):
        with st.status("Analisi flussi video...") as status:
            sig, fps = analyze_video_pro(t_v.name)
            clip = VideoFileClip(t_v.name)
            
            p = {"v_mix":v_mix, "intensity":intensity, "grit":grit, "g_size":g_size, "drone_vol":drone_vol, "seed":seed}
            audio = generate_symbiotic_audio(t_v.name, sig, clip.duration, p)
            
            t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(t_a.name, audio.T, 44100)
            
            out_p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            clip.set_audio(AudioFileClip(t_a.name)).write_videofile(out_p, codec="libx264", audio_codec="aac", logger=None)
            
            status.update(label="Simbiosi Completata!", state="complete")
            st.video(out_p)
            st.download_button("💾 Scarica Video", open(out_p, "rb"), "glitch_pro.mp4")
