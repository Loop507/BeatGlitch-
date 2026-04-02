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
#  ANALISI REATTIVA AI PIXEL E AI CAMBI DI SCENA
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video_reactive(video_path):
    cap = cv2.VideoCapture(video_path)
    energy_curve = []
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Analisi in scala di grigi per velocità
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 75))
        
        if prev_frame is not None:
            # Rileva cambio pixel e varianza (deframmentazione)
            diff = cv2.absdiff(gray, prev_frame)
            score = np.mean(diff) + (np.var(diff) * 0.01)
            energy_curve.append(score)
        else:
            energy_curve.append(0)
        prev_frame = gray
        
    cap.release()
    arr = np.array(energy_curve)
    if len(arr) > 0 and np.max(arr) > 0:
        # Eleviamo a potenza per rendere i glitch "secchi" in sincro con l'immagine
        arr = (arr / np.max(arr)) ** 4
    return arr

# ──────────────────────────────────────────────────────────────────────────────
#  MOTORE AUDIO SINCRO
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_engine(video_path, energy_curve, duration, p, sr=44100):
    np.random.seed(int(p['seed']))
    total_samples = int(duration * sr)
    audio_out = np.zeros((2, total_samples), dtype=np.float32)
    
    # Caricamento audio del video originale
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

    # Scansione ultra-rapida (5ms) per non perdere i frame
    step = 0.005 
    for t_sec in np.arange(0, duration - 0.05, step):
        idx = int(t_sec * sr)
        m_val = e_map[idx]
        
        # Trigger basato sulla sensibilità
        if m_val > (0.1 * (1.1 - p['intensity'])):
            g_len = int(sr * np.random.uniform(0.005, p['g_size']))
            if idx + g_len > total_samples: continue
            
            # Mix tra Audio Video e Audio Processato
            chunk = y_orig[:, idx : idx + g_len].copy()
            
            # STUTTER (micro-ripetizioni tipiche della deframmentazione)
            if m_val > 0.5:
                s_len = int(sr * 0.004)
                if s_len < chunk.shape[1]:
                    chunk = np.tile(chunk[:, :s_len], (1, int(g_len/s_len)+1))[:, :g_len]

            # BITCRUSH
            lev = int(2 + (1 - p['grit']) * 12)
            chunk = np.round(chunk * lev) / lev
            
            env = np.hanning(chunk.shape[1])
            # Applicazione al Mix finale
            audio_out[:, idx:idx+g_len] += chunk * env * p['v_mix'] * m_val

    # Drone di fondo che segue l'energia del video
    t = np.linspace(0, duration, total_samples)
    drone = np.sin(2 * np.pi * 50 * t) * p['drone_vol'] * (0.2 + e_map)
    audio_out += np.tile(drone, (2, 1))

    # Bilanciamento finale: Audio originale + Audio glitchato
    final_mix = (y_orig * p['v_orig_vol']) + (audio_out * 0.8)
    return np.clip(final_mix, -1, 1)

# ──────────────────────────────────────────────────────────────────────────────
#  INTERFACCIA STREAMLIT (TUTTI I COMANDI RIPRISTINATI)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch Studio", layout="wide")

def main():
    st.title("🎬 BeatGlitch: Pixel-Reactive Studio Pro")
    st.markdown("---")

    # Sidebar per i file e il salvataggio
    with st.sidebar:
        st.header("📁 Sorgente & Preset")
        v_file = st.file_uploader("Carica Video (MP4/MOV)", type=["mp4", "mov"])
        
        st.markdown("---")
        st.subheader("💾 Gestione Preset")
        # I valori vengono presi direttamente dagli slider per il download
        if st.button("Genera Link Download Preset"):
            p_data = {
                "v_orig_vol": st.session_state.get('v_orig_vol', 0.5),
                "v_mix": st.session_state.get('v_mix', 1.5),
                "grit": st.session_state.get('grit', 0.8),
                "g_size": st.session_state.get('g_size', 0.1),
                "drone_vol": st.session_state.get('drone_vol', 0.2),
                "intensity": st.session_state.get('intensity', 0.9),
                "seed": st.session_state.get('seed', 42)
            }
            st.download_button("📥 Scarica JSON", json.dumps(p_data), "glitch_preset.json")

    # PANNELLO SLIDER (REINSERITO COMPLETAMENTE)
    st.subheader("🎛️ Pannello di Controllo")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Mix Audio**")
        v_orig_vol = st.slider("Volume Audio Originale", 0.0, 1.0, 0.5, key="v_orig_vol")
        v_mix = st.slider("Intensità Glitch (Effetto)", 0.0, 4.0, 1.5, key="v_mix")
    
    with col2:
        st.write("**Carattere Suono**")
        grit = st.slider("Grit (Bitcrush)", 0.0, 1.0, 0.8, key="grit")
        g_size = st.slider("Durata Micro-Glitch", 0.01, 0.5, 0.1, key="g_size")
    
    with col3:
        st.write("**Reattività & Caos**")
        intensity = st.slider("Sensibilità ai Pixel", 0.0, 1.0, 0.9, key="intensity")
        drone_vol = st.slider("Volume Drone (Fondo)", 0.0, 1.0, 0.2, key="drone_vol")
        seed = st.number_input("Seed (Casualità)", value=42, key="seed")

    # LOGICA DI GENERAZIONE
    if v_file:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        
        if st.button("🚀 GENERA AUDIO REATTIVO AI PIXEL", use_container_width=True):
            try:
                with st.spinner("Analisi differenze frame e deframmentazione..."):
                    energy = analyze_video_reactive(t_v.name)
                    clip = VideoFileClip(t_v.name)
                    
                    params = {
                        "v_orig_vol": v_orig_vol,
                        "v_mix": v_mix, 
                        "grit": grit, 
                        "g_size": g_size, 
                        "intensity": intensity, 
                        "drone_vol": drone_vol,
                        "seed": seed
                    }
                    
                    snd_data = generate_glitch_engine(t_v.name, energy, clip.duration, params)
                    
                    # Salvataggio audio temporaneo
                    t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(t_a.name, snd_data.T, 44100)
                    
                    # Rendering finale
                    final_clip = clip.set_audio(AudioFileClip(t_a.name))
                    out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_f, codec="libx264", audio_codec="aac")
                    
                    st.success("Glitch Art Sincronizzata!")
                    st.video(out_f)
                    with open(out_f, "rb") as f:
                        st.download_button("💾 Scarica Risultato", f, "glitch_art.mp4")
                clip.close()
            except Exception as e:
                st.error(f"Errore: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
