import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# ─────────────────────────────────────────────────────────────────
# 1. ANALISI MICRO-REATTIVA (Pixel-Level Change)
# ─────────────────────────────────────────────────────────────────

def analyze_video_micro(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Segnali per il micro-movimento
    sig = {"lum": [], "mot": [], "hue": [], "flow": []}
    prev_gray = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        img = cv2.resize(frame, (120, 90))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Luminosità e Colore (Macro)
        sig["lum"].append(np.mean(gray) / 255.0)
        sig["hue"].append(np.mean(hsv[:,:,0]) / 180.0)
        
        # 2. Micro-Movimento (Pixel Difference)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            # Rileviamo non solo la media, ma quanto "frigge" l'immagine (std)
            sig["mot"].append(np.mean(diff) / 255.0)
            sig["flow"].append(np.std(diff) / 255.0) 
        else:
            sig["mot"].append(0.0)
            sig["flow"].append(0.0)
        prev_gray = gray
    cap.release()

    # Normalizzazione aggressiva per evitare la monotonia
    for k in sig:
        arr = np.array(sig[k])
        sig[k] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return sig, fps

# ─────────────────────────────────────────────────────────────────
# 2. MOTORE AUDIO: GRANULATORE CAOTICO
# ─────────────────────────────────────────────────────────────────

def generate_nervous_glitch(video_path, sig, duration, p, sr=44100):
    np.random.seed(int(p["seed"]))
    N = int(duration * sr)
    t_ax = np.linspace(0, duration, N)
    
    # Interpolazione segnali
    mot = np.interp(t_ax, np.linspace(0, duration, len(sig["mot"])), sig["mot"])
    flow = np.interp(t_ax, np.linspace(0, duration, len(sig["flow"])), sig["flow"])
    hue = np.interp(t_ax, np.linspace(0, duration, len(sig["hue"])), sig["hue"])
    lum = np.interp(t_ax, np.linspace(0, duration, len(sig["lum"])), sig["lum"])

    audio_out = np.zeros(N)
    
    # --- SORGENTE: Se il video è muto, creiamo un "Digital Hiss" reattivo ---
    try:
        y_src, _ = librosa.load(video_path, sr=sr, mono=True)
        y_src = librosa.util.fix_length(y_src, size=N)
    except:
        # Generiamo un rumore bianco filtrato che "vive" con la luminosità
        y_src = np.random.uniform(-1, 1, N) * lum 

    # --- MICRO-GLITCH ENGINE ---
    # Scansione ogni 4ms per catturare il micro-movimento
    hop = int(sr * 0.004)
    for i in range(0, N - int(sr * p["g_size"]), hop):
        # Il trigger dipende dal mix di movimento e "frittura" pixel (flow)
        trigger = (mot[i] * 0.6 + flow[i] * 0.4) * p["intensity"]
        
        if trigger > 0.1:
            # Lunghezza grano nervosa (più movimento = grani più corti e veloci)
            g_len = int(sr * np.random.uniform(0.002, p["g_size"] * trigger))
            if i + g_len > N: continue
            
            grain = y_src[i : i + g_len].copy()
            
            # Bitcrush dinamico: più colore (hue) = più distorsione
            res = int(2 + (1 - p["grit"]) * 10 * (1 - hue[i]))
            grain = np.round(grain * res) / res
            
            # Applichiamo il volume basato sul trigger
            audio_out[i : i + g_len] += grain * np.hanning(g_len) * p["v_mix"] * trigger

    # --- DRONE DI SUPPORTO (Frequenza basata sul colore) ---
    f_base = 50 + (hue * 200)
    drone = np.sin(2 * np.pi * f_base * t_ax) * p["drone_vol"] * lum
    
    final_mix = np.clip((audio_out * 0.8) + (drone * 0.2), -0.9, 0.9)
    return np.tile(final_mix, (2, 1))

# ─────────────────────────────────────────────────────────────────
# 3. UI STREAMLIT (Ripristinata e corretta)
# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch: Nervous Sync", layout="wide")
st.title("⚡ BeatGlitch: Nervous Audio-Reactivity")

if "seed" not in st.session_state: st.session_state.seed = 42

with st.sidebar:
    st.header("📁 Sorgente")
    v_file = st.file_uploader("Video File", type=["mp4", "mov"])
    st.markdown("---")
    up_json = st.file_uploader("Carica Preset JSON", type="json")
    ld = json.load(up_json) if up_json else {}

st.subheader("🎛️ Micro-Controlli (Simbiotici)")
c1, c2, c3 = st.columns(3)
with c1:
    v_mix = st.slider("Potenza Micro-Glitch", 0.0, 5.0, ld.get("v_mix", 2.5), key="v_mix")
    intensity = st.slider("Sensibilità Reattiva", 0.1, 3.0, ld.get("intensity", 1.5), key="intensity")
with c2:
    grit = st.slider("Grit (Saturazione)", 0.0, 1.0, ld.get("grit", 0.8), key="grit")
    g_size = st.slider("Max Grain Size (s)", 0.01, 0.3, ld.get("g_size", 0.08), key="g_size")
with c3:
    drone_vol = st.slider("Drone di Fondo", 0.0, 1.0, ld.get("drone_vol", 0.1), key="drone_vol")
    seed = st.number_input("Seed", value=int(ld.get("seed", 42)), key="seed")

if v_file:
    t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t_v.write(v_file.read())
    
    if st.button("🚀 GENERA SINCRONIZZAZIONE NERVOSA", use_container_width=True):
        with st.status("Analisi micro-pixel...") as status:
            sig, fps = analyze_video_micro(t_v.name)
            clip = VideoFileClip(t_v.name)
            
            p = {"v_mix":v_mix, "intensity":intensity, "grit":grit, "g_size":g_size, "drone_vol":drone_vol, "seed":seed}
            audio = generate_nervous_glitch(t_v.name, sig, clip.duration, p)
            
            t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(t_a.name, audio.T, 44100)
            
            out_p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            clip.set_audio(AudioFileClip(t_a.name)).write_videofile(out_p, codec="libx264", audio_codec="aac", logger=None)
            
            st.video(out_p)
            st.download_button("💾 Scarica Risultato", open(out_p, "rb"), "glitch_nervous.mp4")
            status.update(label="Sincronizzazione completata!", state="complete")
