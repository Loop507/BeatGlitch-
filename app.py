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
#  ANALISI AVANZATA (COLORE + MOVIMENTO)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video_reactive(video_path):
    cap = cv2.VideoCapture(video_path)
    combined_energy = []
    prev_hist = None
    prev_gray = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Analisi Colore (Istogramma)
        # Questo rileva quando l'immagine cambia colore o si deframmenta
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # 2. Analisi Movimento (Pixel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        
        energy = 0
        if prev_hist is not None:
            # Calcola quanto è cambiato il colore (0 = uguale, 1 = totalmente diverso)
            color_diff = 1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            # Calcola quanto si sono mossi i pixel
            pixel_diff = np.mean(cv2.absdiff(gray, prev_gray)) / 255.0
            # Uniamo le due energie
            energy = color_diff + (pixel_diff * 2.0)
            
        combined_energy.append(energy)
        prev_hist = hist
        prev_gray = gray
        
    cap.release()
    
    arr = np.array(combined_energy)
    if len(arr) > 0 and np.max(arr) > 0:
        # Applichiamo un'enfasi esponenziale per rendere i glitch "esplosivi"
        arr = (arr / np.max(arr)) ** 3 
    else:
        arr = np.zeros(1)
    return arr

# ──────────────────────────────────────────────────────────────────────────────
#  MOTORE AUDIO (ULTRA REATTIVO)
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_engine(video_path, energy_curve, duration, params, sr=44100):
    np.random.seed(int(params['seed']))
    total_samples = int(duration * sr)
    audio_out = np.zeros((2, total_samples), dtype=np.float32)
    
    y_orig, _ = librosa.load(video_path, sr=sr, mono=False)
    if y_orig.ndim == 1: y_orig = np.tile(y_orig, (2, 1))
    if y_orig.shape[1] < total_samples:
        y_orig = np.pad(y_orig, ((0,0), (0, total_samples - y_orig.shape[1])))
    else:
        y_orig = y_orig[:, :total_samples]

    # Curva di energia mappata sui campioni audio
    e_map = np.interp(np.linspace(0, 1, total_samples), 
                      np.linspace(0, 1, len(energy_curve)), 
                      energy_curve).astype(np.float32)

    # Parametri
    step = 0.008 # Risoluzione altissima (8ms)
    
    for t_sec in np.arange(0, duration - 0.05, step):
        idx = int(t_sec * sr)
        current_energy = e_map[idx]
        
        # TRIGGER: se l'energia del colore/movimento supera la soglia
        if current_energy > (0.1 * (1.1 - params['intensity'])):
            
            # Durata basata sull'intensità del glitch visivo
            g_len = int(sr * np.random.uniform(0.005, params['g_size'] * current_energy))
            if idx + g_len > total_samples: continue
            
            chunk = y_orig[:, idx : idx + g_len].copy()
            
            # STUTTER ELETTRICO (il segreto della deframmentazione)
            # Ripetizioni brevissime (3ms) che creano un suono "robotico"
            if current_energy > 0.5:
                s_len = int(sr * 0.003)
                if s_len < chunk.shape[1]:
                    chunk = np.tile(chunk[:, :s_len], (1, int(g_len/s_len)+1))[:, :g_len]

            # DISTORSIONE BITCRUSH (aumenta con l'energia visiva)
            local_grit = params['grit'] * current_energy
            steps = int(2 + (1 - local_grit) * 12)
            chunk = np.round(chunk * steps) / steps
            
            # Applichiamo il suono al mix
            env = np.hanning(chunk.shape[1])
            audio_out[:, idx:idx+g_len] += chunk * env * params['v_mix'] * current_energy

    # Drone di sottofondo che segue l'energia
    t = np.linspace(0, duration, total_samples)
    drone = np.sin(2 * np.pi * 50 * t) * params['drone_vol'] * (0.2 + e_map)
    audio_out += np.tile(drone, (2, 1))

    # Final Limiter
    audio_out = np.clip(audio_out * 1.8, -1, 1)
    return audio_out

# ──────────────────────────────────────────────────────────────────────────────
#  INTERFACCIA
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Glitch Art Reactive", layout="wide")

def main():
    st.title("🧩 Glitch-Sync: Pixel & Color Reactive")
    
    if "drone_vol" not in st.session_state:
        st.session_state.update({
            "drone_vol": 0.2, "v_mix": 1.2, "auto_m": 0.5,
            "grit": 0.8, "g_size": 0.15, "rnd_f": 0.7, "chaos": 0.8, "intensity": 0.9, "seed": 42
        })

    with st.sidebar:
        v_file = st.file_uploader("Carica Video", type=["mp4", "mov"])
        st.markdown("---")
        st.header("💾 Preset")
        curr_p = {k: st.session_state[k] for k in ["drone_vol","v_mix","auto_m","grit","g_size","rnd_f","chaos","intensity","seed"]}
        st.download_button("📥 Salva Preset JSON", json.dumps(curr_p), "preset.json")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.slider("Volume Drone", 0.0, 1.0, key="drone_vol")
        st.slider("Potenza Glitch (Mix)", 0.0, 3.0, key="v_mix")
    with col2:
        st.slider("Grit (Bitcrush)", 0.0, 1.0, key="grit")
        st.slider("Max Durata Grano", 0.01, 0.4, key="g_size")
    with col3:
        st.slider("Chaos Factor", 0.0, 1.0, key="chaos")
        st.slider("Sensibilità Video", 0.0, 1.0, key="intensity")

    if v_file:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        
        if st.button("🚀 GENERA AUDIO REATTIVO AI PIXEL", use_container_width=True):
            with st.spinner("Analisi Istogrammi Colore e Deframmentazione..."):
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
                st.download_button("💾 Scarica", open(out_f, "rb"), "glitch_sync.mp4")

if __name__ == "__main__":
    main()
