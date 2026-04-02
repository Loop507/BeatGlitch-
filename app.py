import streamlit as st
import os
import numpy as np
import tempfile
import traceback
from moviepy.editor import VideoFileClip, AudioFileClip
import librosa
import soundfile as sf
import json
import cv2 

# ──────────────────────────────────────────────────────────────────────────────
#  PRESETS DI FABBRICA
# ──────────────────────────────────────────────────────────────────────────────

PRESETS_FACTORY = {
    "🧩 Pixel Reactive (G4)": {
        "drone_vol": 0.2, "v_mix": 1.5, "auto_m": 0.5,
        "grit": 0.85, "g_size": 0.15, "rnd_f": 0.8, "chaos": 0.8, "intensity": 0.9, "seed": 42
    },
    "🔥 Hyper-Glitch": {
        "drone_vol": 0.1, "v_mix": 2.5, "auto_m": 0.2,
        "grit": 0.95, "g_size": 0.05, "rnd_f": 0.9, "chaos": 0.9, "intensity": 1.0, "seed": 666
    },
    "🌌 Ambient Drone": {
        "drone_vol": 0.7, "v_mix": 0.4, "auto_m": 0.8,
        "grit": 0.2, "g_size": 0.8, "rnd_f": 0.2, "chaos": 0.3, "intensity": 0.5, "seed": 777
    }
}

# ──────────────────────────────────────────────────────────────────────────────
#  ANALISI REATTIVA (COLORE + PIXEL)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video_reactive(video_path):
    cap = cv2.VideoCapture(video_path)
    combined_energy = []
    prev_hist = None
    prev_gray = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Analisi Colore (Rileva deframmentazione e cambi tono)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # 2. Analisi Movimento Pixel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        
        energy = 0
        if prev_hist is not None:
            color_diff = 1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            pixel_diff = np.mean(cv2.absdiff(gray, prev_gray)) / 255.0
            energy = color_diff + (pixel_diff * 2.5)
            
        combined_energy.append(energy)
        prev_hist = hist
        prev_gray = gray
        
    cap.release()
    arr = np.array(combined_energy)
    if len(arr) > 0 and np.max(arr) > 0:
        arr = (arr / np.max(arr)) ** 3 # Esponenziale per isolare i glitch
    else:
        arr = np.zeros(1)
    return arr

# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_engine(video_path, energy_curve, duration, params, sr=44100):
    np.random.seed(int(params['seed']))
    total_samples = int(duration * sr)
    audio_out = np.zeros((2, total_samples), dtype=np.float32)
    
    try:
        y_orig, _ = librosa.load(video_path, sr=sr, mono=False)
        if y_orig.ndim == 1: y_orig = np.tile(y_orig, (2, 1))
        if y_orig.shape[1] < total_samples:
            y_orig = np.pad(y_orig, ((0,0), (0, total_samples - y_orig.shape[1])))
        else:
            y_orig = y_orig[:, :total_samples]
    except:
        y_orig = np.zeros((2, total_samples), dtype=np.float32)

    e_map = np.interp(np.linspace(0, 1, total_samples), 
                      np.linspace(0, 1, len(energy_curve)), 
                      energy_curve).astype(np.float32)

    step = 0.008 
    for t_sec in np.arange(0, duration - 0.05, step):
        idx = int(t_sec * sr)
        current_energy = e_map[idx]
        
        # Trigger basato su sensibilità
        if current_energy > (0.1 * (1.1 - params['intensity'])):
            g_dur = np.random.uniform(0.005, params['g_size'] * current_energy)
            g_len = int(sr * g_dur)
            if idx + g_len > total_samples: continue
            
            chunk = y_orig[:, idx : idx + g_len].copy()
            
            # Effetto Stutter (Micro-ripetizioni)
            if current_energy > 0.4:
                s_len = int(sr * 0.004)
                if s_len < chunk.shape[1]:
                    chunk = np.tile(chunk[:, :s_len], (1, int(g_len/s_len)+1))[:, :g_len]

            # Bitcrush
            lev = int(2 + (1 - params['grit']) * 12)
            chunk = np.round(chunk * lev) / lev
            
            env = np.hanning(chunk.shape[1])
            audio_out[:, idx:idx+g_len] += chunk * env * params['v_mix'] * current_energy

    # Drone modulato
    t = np.linspace(0, duration, total_samples)
    freq = 45 + (params['auto_m'] * 40)
    drone = np.sin(2 * np.pi * freq * t) * params['drone_vol'] * (0.3 + e_map)
    audio_out += np.tile(drone, (2, 1))

    return np.clip(audio_out * 1.8, -1, 1)

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN APP UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch Studio", layout="wide")

def main():
    st.title("🎬 BeatGlitch: Pixel-Sync Studio")
    st.write("Il suono reagisce alla deframmentazione e al colore del video.")

    # 1. Inizializzazione Session State
    if "drone_vol" not in st.session_state:
        for k, v in PRESETS_FACTORY["🧩 Pixel Reactive (G4)"].items():
            st.session_state[k] = v

    # SIDEBAR
    with st.sidebar:
        st.header("📁 File & Preset")
        v_file = st.file_uploader("Carica Video", type=["mp4", "mov"])
        
        st.markdown("---")
        st.subheader("🎭 Preset Veloci")
        for name, p_vals in PRESETS_FACTORY.items():
            if st.button(name, use_container_width=True):
                for k, v in p_vals.items(): st.session_state[k] = v
                st.rerun()

        st.markdown("---")
        st.subheader("💾 Gestione JSON")
        # Download
        curr_p = {k: st.session_state[k] for k in PRESETS_FACTORY["🧩 Pixel Reactive (G4)"].keys()}
        st.download_button("📥 Scarica Preset", json.dumps(curr_p), "preset.json", use_container_width=True)
        # Upload
        up_json = st.file_uploader("📤 Carica Preset", type="json")
        if up_json:
            if st.button("Applica"):
                d = json.load(up_json)
                for k, v in d.items(): st.session_state[k] = v
                st.rerun()

    # GLI SLIDER (Ripristinati qui)
    st.subheader("🎛️ Pannello di Controllo")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.slider("Volume Drone", 0.0, 1.0, key="drone_vol")
        st.slider("Potenza Glitch (Mix)", 0.0, 4.0, key="v_mix")
        st.slider("Auto-Morphing", 0.0, 1.0, key="auto_m")
    with c2:
        st.slider("Grit (Bitcrush)", 0.0, 1.0, key="grit")
        st.slider("Max Durata Grano", 0.01, 0.6, key="g_size")
        st.slider("Random Factor", 0.0, 1.0, key="rnd_f")
    with c3:
        st.slider("Chaos Factor", 0.0, 1.0, key="chaos")
        st.slider("Sensibilità Video", 0.0, 1.0, key="intensity")
        st.number_input("Seed", key="seed")

    # LOGICA DI GENERAZIONE
    if v_file:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        
        if st.button("🚀 GENERA AUDIO REATTIVO AI PIXEL", use_container_width=True):
            try:
                with st.spinner("Analisi Istogrammi e Pixel in corso..."):
                    energy_curve = analyze_video_reactive(t_v.name)
                    clip = VideoFileClip(t_v.name)
                    
                    p_dict = {k: st.session_state[k] for k in ["drone_vol","v_mix","auto_m","grit","g_size","rnd_f","chaos","intensity","seed"]}
                    snd = generate_glitch_engine(t_v.name, energy_curve, clip.duration, p_dict)
                    
                    t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(t_a.name, snd.T, 44100)
                    
                    final_clip = clip.set_audio(AudioFileClip(t_a.name))
                    out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_f, codec="libx264")
                    
                    st.video(out_f)
                    st.download_button("💾 Scarica Risultato", open(out_f, "rb"), "glitch_sync.mp4")
                clip.close()
            except Exception as e:
                st.error(f"Errore: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
