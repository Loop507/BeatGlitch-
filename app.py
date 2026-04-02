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
#  FUNZIONE DI ANALISI (IL "CERVELLO" REATTIVO)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video_vibrancy(video_path):
    cap = cv2.VideoCapture(video_path)
    energy_curve = []
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Convertiamo in scala di grigi e riduciamo per velocità
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (80, 60))
        
        if prev_frame is not None:
            # 1. Differenza di pixel (movimento)
            diff = cv2.absdiff(gray, prev_frame)
            # 2. Varianza (rileva se l'immagine si "rompe" o deframmenta)
            variance = np.var(diff)
            # 3. Cambio luce (flash)
            light_change = abs(np.mean(gray) - np.mean(prev_frame))
            
            # Uniamo i fattori per un punteggio di "Glitch visivo"
            score = np.mean(diff) + (light_change * 1.5) + (variance * 0.05)
            energy_curve.append(score)
        else:
            energy_curve.append(0)
        prev_frame = gray
        
    cap.release()
    arr = np.array(energy_curve)
    if len(arr) > 0 and np.max(arr) > 0:
        # Eleviamo a potenza per rendere il suono "secco" e reattivo solo ai picchi
        arr = (arr / np.max(arr)) ** 4
    return arr

# ──────────────────────────────────────────────────────────────────────────────
#  MOTORE AUDIO GLITCH
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_engine(video_path, energy_curve, duration, p, sr=44100):
    np.random.seed(int(p['seed']))
    total_samples = int(duration * sr)
    audio_out = np.zeros((2, total_samples), dtype=np.float32)
    
    # Carica audio originale
    y_orig, _ = librosa.load(video_path, sr=sr, mono=False)
    if y_orig.ndim == 1: y_orig = np.tile(y_orig, (2, 1))
    if y_orig.shape[1] < total_samples:
        y_orig = np.pad(y_orig, ((0,0), (0, total_samples - y_orig.shape[1])))
    else:
        y_orig = y_orig[:, :total_samples]

    e_map = np.interp(np.linspace(0, 1, total_samples), 
                      np.linspace(0, 1, len(energy_curve)), 
                      energy_curve).astype(np.float32)

    # Scansione ultra-veloce (5ms)
    step = 0.005 
    for t_sec in np.arange(0, duration - 0.05, step):
        idx = int(t_sec * sr)
        m_val = e_map[idx]
        
        # Trigger basato sulla sensibilità (Intensity)
        if m_val > (0.05 * (1.1 - p['intensity'])):
            # Durata del glitch proporzionale all'energia visiva
            g_dur = np.random.uniform(0.005, p['g_size'] * (0.2 + m_val))
            g_len = int(sr * g_dur)
            if idx + g_len > total_samples: continue
            
            chunk = y_orig[:, idx : idx + g_len].copy()
            
            # STUTTER (Ripetizione robotica)
            s_len = int(sr * 0.003)
            if s_len < chunk.shape[1]:
                chunk = np.tile(chunk[:, :s_len], (1, int(g_len/s_len)+1))[:, :g_len]

            # BITCRUSH (Distorsione)
            lev = int(2 + (1 - p['grit']) * 12)
            chunk = np.round(chunk * lev) / lev
            
            # Inviluppo e mix
            env = np.hanning(chunk.shape[1])
            audio_out[:, idx:idx+g_len] += chunk * env * p['v_mix'] * m_val

    # Drone di fondo che segue il movimento
    t = np.linspace(0, duration, total_samples)
    drone = np.sin(2 * np.pi * 50 * t) * p['drone_vol'] * (0.3 + e_map)
    audio_out += np.tile(drone, (2, 1))

    return np.clip(audio_out * 2.0, -1, 1)

# ──────────────────────────────────────────────────────────────────────────────
#  INTERFACCIA UTENTE (UI)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch V3", layout="wide")

def main():
    st.title("🎬 BeatGlitch: Pixel-Reactivity V3")
    st.write("L'audio reagisce ai cambi di frame, pixel e deframmentazione.")

    # SIDEBAR
    with st.sidebar:
        st.header("📁 Sorgente Video")
        v_file = st.file_uploader("Carica Video (MP4/MOV)", type=["mp4", "mov"])
        
        st.markdown("---")
        st.header("💾 Preset & Backup")
        # Invece di tasti che rompono gli slider, usiamo un download manuale
        st.info("Regola gli slider e poi clicca Genera.")

    # PANNELLO SLIDER (Sempre visibili)
    st.subheader("🎛️ Parametri di Sincronizzazione")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        v_mix = st.slider("Potenza Glitch (Volume)", 0.0, 5.0, 1.5)
        intensity = st.slider("Sensibilità Video (Sincro)", 0.0, 1.0, 0.9)
    with col2:
        grit = st.slider("Grit (Bitcrush)", 0.0, 1.0, 0.8)
        g_size = st.slider("Durata Glitch", 0.01, 0.5, 0.1)
    with col3:
        drone_vol = st.slider("Volume Drone", 0.0, 1.0, 0.2)
        seed = st.number_input("Seed (Casualità)", value=42)

    # GENERAZIONE
    if v_file:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        
        if st.button("🚀 GENERA AUDIO REATTIVO", use_container_width=True):
            try:
                with st.spinner("Analisi differenze pixel e deframmentazione..."):
                    energy = analyze_video_vibrancy(t_v.name)
                    clip = VideoFileClip(t_v.name)
                    
                    params = {
                        "v_mix": v_mix, 
                        "grit": grit, 
                        "g_size": g_size, 
                        "intensity": intensity, 
                        "drone_vol": drone_vol,
                        "seed": seed
                    }
                    
                    snd = generate_glitch_engine(t_v.name, energy, clip.duration, params)
                    
                    # Salvataggio temporaneo
                    t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(t_a.name, snd.T, 44100)
                    
                    # Merge Audio/Video
                    final_clip = clip.set_audio(AudioFileClip(t_a.name))
                    out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_f, codec="libx264")
                    
                    st.video(out_f)
                    st.success("Glitch Art Generata con successo!")
                    with open(out_f, "rb") as f:
                        st.download_button("💾 Scarica Risultato", f, "glitch_art.mp4")
                clip.close()
            except Exception as e:
                st.error(f"Errore tecnico: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
