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
#  PRESET ULTRA-REATTIVO (OTTIMIZZATO PER G4)
# ──────────────────────────────────────────────────────────────────────────────

PRESETS_FACTORY = {
    "🔥 Hyper-Reactive Glitch": {
        "drone_vol": 0.2, "v_mix": 1.0, "auto_m": 0.5,
        "grit": 0.9, "g_size": 0.05, "rnd_f": 0.9, "chaos": 0.9, "intensity": 1.0, "seed": 42
    },
    "👾 Hard Glitch": {
        "drone_vol": 0.2, "v_mix": 0.7, "auto_m": 0.1,
        "grit": 0.9, "g_size": 0.05, "rnd_f": 0.8, "chaos": 0.8, "intensity": 0.9, "seed": 666
    }
}

# ──────────────────────────────────────────────────────────────────────────────
#  ANALISI MOVIMENTO POTENZIATA
# ──────────────────────────────────────────────────────────────────────────────

def analyze_motion_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    motion_data = []
    prev_gray = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_small = cv2.resize(frame, (120, 90))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.mean(diff)
            motion_data.append(motion_score)
        else:
            motion_data.append(0)
        prev_gray = gray
    cap.release()
    
    motion_array = np.array(motion_data)
    if len(motion_array) > 0 and np.max(motion_array) > 0:
        # Enfatizziamo i picchi (elevando al quadrato) per isolare i glitch
        motion_array = (motion_array / np.max(motion_array)) ** 2
    else:
        motion_array = np.zeros(1)
    return motion_array

# ──────────────────────────────────────────────────────────────────────────────
#  MOTORE AUDIO "NEVROTICO"
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_engine(video_path, motion_curve, duration, sr=44100, 
                          d_vol=0.3, v_mix=0.5, grit=0.6, 
                          chaos=0.5, g_size=0.4, 
                          rnd_f=0.5, auto_m=0.4, intensity=0.7, seed=42):
    
    np.random.seed(int(seed))
    total_samples = int(duration * sr)
    audio_out = np.zeros((2, total_samples), dtype=np.float32)
    
    try:
        y_orig, _ = librosa.load(video_path, sr=sr, mono=False)
        if y_orig.ndim == 1: y_orig = np.tile(y_orig, (2, 1))
        if y_orig.shape[1] > total_samples: y_orig = y_orig[:, :total_samples]
        else: y_orig = np.pad(y_orig, ((0,0), (0, total_samples - y_orig.shape[1])))
    except:
        y_orig = np.zeros((2, total_samples), dtype=np.float32)

    motion_map = np.interp(np.linspace(0, 1, total_samples), 
                           np.linspace(0, 1, len(motion_curve)), 
                           motion_curve).astype(np.float32)

    # 1. DRONE (Molto basso, solo come texture)
    t = np.linspace(0, duration, total_samples, dtype=np.float32)
    drone = np.sin(2 * np.pi * 40 * t) * d_vol * (1.0 - motion_map * 0.5)
    audio_out += np.tile(drone, (2, 1))

    # 2. GLITCH CHIRURGICO (Passo ridotto a 0.01 per micro-sincronia)
    step = 0.01 
    for t_sec in np.arange(0, duration - 0.1, step):
        idx = int(t_sec * sr)
        m_val = motion_map[idx]
        
        # SOGLIA DI TRIGGER: solo se il movimento è forte o per puro caos
        if m_val > (0.6 * (1.1 - intensity)) or np.random.random() < (rnd_f * 0.05):
            
            # Grani cortissimi per fare "click" e "pop"
            dur = np.random.uniform(0.01, g_size)
            g_len = int(sr * dur)
            if idx + g_len > total_samples: continue
            
            chunk = y_orig[:, idx : idx + g_len].copy()
            
            # STUTTER AGGRESSIVO sui picchi di movimento
            if m_val > 0.4:
                sub_len = int(sr * 0.005) # 5ms
                if sub_len < chunk.shape[1]:
                    stutter = np.tile(chunk[:, :sub_len], (1, int(g_len/sub_len) + 1))
                    chunk = stutter[:, :g_len]

            # BITCRUSH estremo legato al movimento
            local_grit = grit * (0.7 + m_val * 0.3)
            steps = int(2 + (1 - local_grit) * 8)
            chunk = np.round(chunk * steps) / steps
            
            # Spatial Pan casuale
            pan = np.random.uniform(0.2, 0.8)
            env = np.hanning(g_len) # Inviluppo più secco
            
            audio_out[0, idx:idx+g_len] += chunk[0] * env * v_mix * m_val * pan
            audio_out[1, idx:idx+g_len] += chunk[1] * env * v_mix * m_val * (1-pan)

    audio_out = np.clip(audio_out * (1.5 + intensity), -1, 1)
    return audio_out

# ──────────────────────────────────────────────────────────────────────────────
#  UI (Semplificata per testare subito)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch Hyper-Sync", layout="wide")

def main():
    st.title("🎬 BeatGlitch: Hyper-Reactive Sync")
    st.write("Ottimizzato per glitch rapidi e audio-reactive art.")

    if "drone_vol" not in st.session_state:
        for k, v in PRESETS_FACTORY["🔥 Hyper-Reactive Glitch"].items():
            st.session_state[k] = v

    with st.sidebar:
        v_file = st.file_uploader("Carica Video", type=["mp4", "mov"])
        for name, p_vals in PRESETS_FACTORY.items():
            if st.button(name, use_container_width=True):
                for k, v in p_vals.items(): st.session_state[k] = v
                st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.slider("Volume Drone", 0.0, 1.0, key="drone_vol")
        st.slider("Mix Audio Video", 0.0, 1.0, key="v_mix")
    with c2:
        st.slider("Grit (Distorsione)", 0.0, 1.0, key="grit")
        st.slider("Dimensione Grano", 0.01, 0.3, key="g_size")
    with c3:
        st.slider("Chaos Factor", 0.0, 1.0, key="chaos")
        st.slider("Intensità Reazione", 0.0, 1.0, key="intensity")
        st.number_input("Seed", key="seed")

    if v_file:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        
        if st.button("🚀 GENERA GLITCH ART SINCRO", use_container_width=True):
            with st.spinner("Analisi sub-frame dei pixel..."):
                m_curve = analyze_motion_optical_flow(t_v.name)
                clip = VideoFileClip(t_v.name)
                snd = generate_glitch_engine(t_v.name, m_curve, clip.duration, **{k: st.session_state[k] for k in ["drone_vol","v_mix","grit","g_size","chaos","intensity","seed"]})
                
                t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(t_a.name, snd.T, 44100)
                final_clip = clip.set_audio(AudioFileClip(t_a.name))
                out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                final_clip.write_videofile(out_f, codec="libx264")
                st.video(out_f)
                st.download_button("💾 Scarica", open(out_f, "rb"), "glitch.mp4")

if __name__ == "__main__":
    main()
