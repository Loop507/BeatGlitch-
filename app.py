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
#  ANALISI REATTIVA (VARIANZA E LUMINOSITÀ) - OTTIMIZZATA PER G4
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video_vibrancy(video_path):
    cap = cv2.VideoCapture(video_path)
    energy_curve = []
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Ridimensiona e converti in scala di grigi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (80, 60))
        
        if prev_frame is not None:
            # Rileva la differenza di pixel (movimento)
            diff = cv2.absdiff(gray, prev_frame)
            # Rileva il cambio di luminosità totale (flash/scena)
            brightness_change = abs(np.mean(gray) - np.mean(prev_frame))
            # Forza del glitch = (cambio pixel * cambio luce)
            score = np.mean(diff) + (brightness_change * 2.0)
            energy_curve.append(score)
        else:
            energy_curve.append(0)
        prev_frame = gray
        
    cap.release()
    arr = np.array(energy_curve)
    if len(arr) > 0 and np.max(arr) > 0:
        # Esponente 4.0: rende il silenzio totale finché non c'è un glitch visivo
        arr = (arr / np.max(arr)) ** 4
    return arr

# ──────────────────────────────────────────────────────────────────────────────
#  MOTORE AUDIO
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_engine(video_path, energy_curve, duration, p, sr=44100):
    np.random.seed(int(p['seed']))
    total_samples = int(duration * sr)
    audio_out = np.zeros((2, total_samples), dtype=np.float32)
    
    y_orig, _ = librosa.load(video_path, sr=sr, mono=False)
    if y_orig.ndim == 1: y_orig = np.tile(y_orig, (2, 1))
    if y_orig.shape[1] < total_samples:
        y_orig = np.pad(y_orig, ((0,0), (0, total_samples - y_orig.shape[1])))
    else:
        y_orig = y_orig[:, :total_samples]

    e_map = np.interp(np.linspace(0, 1, total_samples), 
                      np.linspace(0, 1, len(energy_curve)), 
                      energy_curve).astype(np.float32)

    # Scansione ogni 5ms per una precisione totale
    step = 0.005 
    for t_sec in np.arange(0, duration - 0.05, step):
        idx = int(t_sec * sr)
        m_val = e_map[idx]
        
        # Trigger basato su sensibilità estrema
        if m_val > (0.05 * (1.1 - p['intensity'])):
            g_dur = np.random.uniform(0.005, p['g_size'] * (0.1 + m_val))
            g_len = int(sr * g_dur)
            if idx + g_len > total_samples: continue
            
            chunk = y_orig[:, idx : idx + g_len].copy()
            
            # STUTTER ELETTRICO
            s_len = int(sr * 0.003)
            if s_len < chunk.shape[1]:
                chunk = np.tile(chunk[:, :s_len], (1, int(g_len/s_len)+1))[:, :g_len]

            # BITCRUSH dinamico
            lev = int(2 + (1 - p['grit']) * 12)
            chunk = np.round(chunk * lev) / lev
            
            env = np.hanning(chunk.shape[1])
            audio_out[:, idx:idx+g_len] += chunk * env * p['v_mix'] * m_val

    return np.clip(audio_out * 2.0, -1, 1)

# ──────────────────────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch V3", layout="wide")

def main():
    st.title("🎬 BeatGlitch: Pixel-Reactivity V3")

    # FIX Errore SessionState: definiamo i valori di default prima degli slider
    defaults = {"drone_vol": 0.2, "v_mix": 1.5, "grit": 0.8, "g_size": 0.1, "intensity": 0.9, "seed": 42}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        v_file = st.file_uploader("Carica Video", type=["mp4", "mov"])
        st.markdown("---")
        if st.button("RESET PARAMETRI"):
            for k, v in defaults.items(): st.session_state[k] = v
            st.rerun()

    st.subheader("🎛️ Controllo Reattività")
    c1, c2, c3 = st.columns(3)
    with c1:
        v_mix = st.slider("Potenza Glitch", 0.0, 5.0, key="v_mix")
        intensity = st.slider("Sensibilità Video (Sincro)", 0.0, 1.0, key="intensity")
    with c2:
        grit = st.slider("Grit (Bitcrush)", 0.0, 1.0, key="grit")
        g_size = st.slider("Durata Glitch", 0.01, 0.5, key="g_size")
    with c3:
        seed = st.number_input("Seed", value=42, key="seed")
        st.info("Aumenta 'Sensibilità' se l'audio non reagisce abbastanza.")

    if v_file:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        
        if st.button("🚀 GENERA AUDIO REATTIVO", use_container_width=True):
            try:
                with st.spinner("Analisi differenze pixel e luce..."):
                    energy = analyze_video_vibrancy(t_v.name)
                    clip = VideoFileClip(t_v.name)
                    
                    params = {"v_mix": v_mix, "grit": grit, "g_size": g_size, "intensity": intensity, "seed": seed}
                    snd = generate_glitch_engine(t_v.name, energy, clip.duration, params)
                    
                    t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(t_a.name, snd.T, 44100)
                    final_clip = clip.set_audio(AudioFileClip(t_a.name))
                    out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_f, codec="libx264")
                    
                    st.video(out_f)
                    st.success("Sincronizzazione completata su ogni pixel!")
            except Exception as e:
                st.error(f"Errore: {e}")

if __name__ == "__main__":
    main()
