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
#  ANALISI VIDEO: COLORE + LUMINOSITÀ + PIXEL
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video_v3(video_path):
    cap = cv2.VideoCapture(video_path)
    energy_curve = []
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Ridimensiona per velocità
        small_frame = cv2.resize(frame, (60, 45))
        
        if prev_frame is not None:
            # 1. Differenza di colore (RGB)
            diff_color = np.mean(cv2.absdiff(small_frame, prev_frame))
            # 2. Differenza di luminosità
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff_brightness = abs(np.mean(gray) - np.mean(prev_gray))
            
            # Score combinato (molto sensibile ai cambi di immagine)
            score = diff_color + (diff_brightness * 2.0)
            energy_curve.append(score)
        else:
            energy_curve.append(0)
        prev_frame = small_frame
        
    cap.release()
    arr = np.array(energy_curve)
    if len(arr) > 0 and np.max(arr) > 0:
        # Eleviamo a 5 per isolare solo i cambiamenti netti (glitch/tagli)
        arr = (arr / np.max(arr)) ** 5
    return arr

# ──────────────────────────────────────────────────────────────────────────────
#  MOTORE AUDIO
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_v3(video_path, energy, duration, p, sr=44100):
    np.random.seed(int(p['seed']))
    total_samples = int(duration * sr)
    audio_out = np.zeros((2, total_samples), dtype=np.float32)
    
    # Audio Originale
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
                      np.linspace(0, 1, len(energy)), energy).astype(np.float32)

    # Scansione sub-frame (4ms)
    step = 0.004
    for t_sec in np.arange(0, duration - 0.05, step):
        idx = int(t_sec * sr)
        m_val = e_map[idx]
        
        # Trigger se il video cambia o deframmenta
        if m_val > (0.05 * (1.1 - p['intensity'])):
            g_len = int(sr * np.random.uniform(0.005, p['g_size']))
            if idx + g_len > total_samples: continue
            
            chunk = y_orig[:, idx : idx + g_len].copy()
            
            # Stutter (Effetto Robotico)
            if m_val > 0.4:
                s_len = int(sr * 0.003)
                if s_len < chunk.shape[1]:
                    chunk = np.tile(chunk[:, :s_len], (1, int(g_len/s_len)+1))[:, :g_len]

            # Bitcrush
            lev = int(2 + (1 - p['grit']) * 15)
            chunk = np.round(chunk * lev) / lev
            
            env = np.hanning(chunk.shape[1])
            audio_out[:, idx:idx+g_len] += chunk * env * p['v_mix'] * m_val

    # Fondo Drone
    t = np.linspace(0, duration, total_samples)
    drone = np.sin(2 * np.pi * 45 * t) * p['drone_vol'] * (0.1 + e_map)
    audio_out += np.tile(drone, (2, 1))

    final = (y_orig * p['v_orig_vol']) + (audio_out * 1.0)
    return np.clip(final, -1, 1)

# ──────────────────────────────────────────────────────────────────────────────
#  INTERFACCIA
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch V3", layout="wide")

def main():
    st.title("🎬 BeatGlitch: Hyper-Sync Pro")
    
    # Inizializzazione Session State sicura
    if 'p_seed' not in st.session_state: st.session_state.p_seed = 42

    with st.sidebar:
        st.header("📁 File & Preset")
        v_file = st.file_uploader("Carica Video", type=["mp4", "mov"])
        
        st.markdown("---")
        up_json = st.file_uploader("📤 Carica Preset JSON", type="json")
        if up_json:
            if st.button("APPLICA PRESET CARICATO"):
                d = json.load(up_json)
                # Salviamo i dati per usarli dopo
                st.session_state.loaded_params = d
                st.success("Dati pronti, regola gli slider se necessario.")

        if st.button("📥 Prepara Download Preset"):
            p_now = {
                "v_orig_vol": st.session_state.v_orig_vol,
                "v_mix": st.session_state.v_mix,
                "grit": st.session_state.grit,
                "g_size": st.session_state.g_size,
                "intensity": st.session_state.intensity,
                "drone_vol": st.session_state.drone_vol,
                "seed": st.session_state.p_seed
            }
            st.download_button("Scarica Ora", json.dumps(p_now), "glitch_preset.json")

    st.subheader("🎛️ Pannello di Controllo")
    c1, c2, c3 = st.columns(3)
    
    # Valori di default o caricati
    ld = st.session_state.get('loaded_params', {})

    with c1:
        v_orig_vol = st.slider("Volume Originale", 0.0, 1.0, ld.get("v_orig_vol", 0.5), key="v_orig_vol")
        v_mix = st.slider("Potenza Glitch", 0.0, 5.0, ld.get("v_mix", 2.0), key="v_mix")
    with c2:
        grit = st.slider("Grit (Distorsione)", 0.0, 1.0, ld.get("grit", 0.8), key="grit")
        g_size = st.slider("Durata Glitch", 0.01, 0.5, ld.get("g_size", 0.12), key="g_size")
    with c3:
        intensity = st.slider("Sensibilità Video", 0.0, 1.0, ld.get("intensity", 0.9), key="intensity")
        drone_vol = st.slider("Volume Drone", 0.0, 1.0, ld.get("drone_vol", 0.2), key="drone_vol")
        seed = st.number_input("Seed", value=int(ld.get("seed", 42)), key="p_seed")

    if v_file:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        
        if st.button("🚀 GENERA AUDIO REATTIVO AI PIXEL", use_container_width=True):
            try:
                with st.spinner("Analisi Colore e Deframmentazione..."):
                    energy = analyze_video_v3(t_v.name)
                    clip = VideoFileClip(t_v.name)
                    
                    params = {
                        "v_orig_vol": v_orig_vol, "v_mix": v_mix, "grit": grit,
                        "g_size": g_size, "intensity": intensity, 
                        "drone_vol": drone_vol, "seed": seed
                    }
                    
                    snd = generate_glitch_v3(t_v.name, energy, clip.duration, params)
                    
                    t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(t_a.name, snd.T, 44100)
                    
                    final_clip = clip.set_audio(AudioFileClip(t_a.name))
                    out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_f, codec="libx264")
                    
                    st.video(out_f)
                    st.success("Sincronizzazione completata!")
            except Exception as e:
                st.error(f"Errore: {e}")

if __name__ == "__main__":
    main()
