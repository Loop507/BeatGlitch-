import streamlit as st
import os
import numpy as np
import tempfile
import cv2
import librosa
import soundfile as sf
import json
from moviepy.editor import VideoFileClip, AudioFileClip

# ──────────────────────────────────────────────────────────────────────────────
#  ANALISI ULTRA-SENSIBILE (PIXEL-PER-PIXEL)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video_v4(video_path):
    cap = cv2.VideoCapture(video_path)
    raw_scores = []
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Analisi ad alta risoluzione (160x120) per catturare piccoli glitch
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, (160, 120))
        
        if prev_frame is not None:
            # Differenza assoluta tra i frame
            diff = cv2.absdiff(curr_gray, prev_frame)
            # Calcolo energia: media dei pixel cambiati + deviazione standard (caos)
            score = np.mean(diff) + np.std(diff)
            raw_scores.append(score)
        else:
            raw_scores.append(0)
        prev_frame = curr_gray
        
    cap.release()
    
    scores = np.array(raw_scores)
    if len(scores) > 0 and np.max(scores) > 0:
        # Normalizzazione: il punto più "movimentato" del video diventa 1.0
        scores = scores / np.max(scores)
        # Applichiamo una curva di risposta più naturale (esponenziale morbida)
        scores = np.power(scores, 1.5)
    else:
        scores = np.ones(1) * 0.1 # Fallback se il video è statico
    return scores

# ──────────────────────────────────────────────────────────────────────────────
#  MOTORE AUDIO REATTIVO
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_v4(video_path, energy_arr, duration, p, sr=44100):
    np.random.seed(int(p['seed']))
    total_samples = int(duration * sr)
    
    # Caricamento Audio Sorgente
    try:
        y_src, _ = librosa.load(video_path, sr=sr, mono=False)
        if y_src.ndim == 1: y_src = np.tile(y_src, (2, 1))
        # Trim o Pad per matchare la durata
        if y_src.shape[1] < total_samples:
            y_src = np.pad(y_src, ((0,0), (0, total_samples - y_src.shape[1])))
        else:
            y_src = y_src[:, :total_samples]
    except:
        y_src = np.zeros((2, total_samples), dtype=np.float32)

    # Mapping energia video su campioni audio
    e_map = np.interp(np.linspace(0, 1, total_samples), 
                      np.linspace(0, 1, len(energy_arr)), 
                      energy_arr).astype(np.float32)

    audio_out = np.zeros((2, total_samples), dtype=np.float32)
    
    # Sintesi Glitch Granulare
    grain_step = 0.005 # 5ms
    for t in np.arange(0, duration - 0.05, grain_step):
        idx = int(t * sr)
        # Forza del glitch in questo istante
        power = e_map[idx] * p['intensity']
        
        if power > 0.02 or np.random.random() < 0.05:
            # Lunghezza grano influenzata dal video
            g_len = int(sr * np.random.uniform(0.01, p['g_size']))
            if idx + g_len >= total_samples: continue
            
            chunk = y_src[:, idx : idx + g_len].copy()
            
            # Bitcrush (Grit)
            if p['grit'] > 0:
                steps = int(2 + (1.0 - p['grit']) * 16)
                chunk = np.round(chunk * steps) / steps
            
            # Applica Inviluppo e Volume Glitch
            env = np.hanning(g_len)
            audio_out[:, idx:idx+g_len] += chunk * env * p['v_mix'] * (power + 0.1)

    # Drone di sottofondo (sempre presente ma modulato dal video)
    time_axis = np.linspace(0, duration, total_samples)
    drone = np.sin(2 * np.pi * 55 * time_axis) * p['drone_vol'] * (0.2 + e_map)
    audio_out += np.tile(drone, (2, 1))

    # Mix Finale: Audio Originale + Glitch
    final_mix = (y_src * p['v_orig_vol']) + (audio_out * 0.7)
    return np.clip(final_mix, -1, 1)

# ──────────────────────────────────────────────────────────────────────────────
#  INTERFACCIA STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch Studio V4", layout="wide")

def main():
    st.title("🎬 BeatGlitch: Pixel-Reactivity V4")
    
    # Sidebar: File e Preset
    with st.sidebar:
        st.header("📂 Risorse")
        v_file = st.file_uploader("Carica Video", type=["mp4", "mov"])
        
        st.markdown("---")
        st.subheader("💾 Preset JSON")
        up_json = st.file_uploader("Carica Preset", type="json")
        
        # Download Preset attuale
        if st.button("Prepara Download Preset"):
            curr = {
                "v_orig_vol": st.session_state.get("v_orig_vol", 0.3),
                "v_mix": st.session_state.get("v_mix", 1.5),
                "grit": st.session_state.get("grit", 0.7),
                "g_size": st.session_state.get("g_size", 0.1),
                "intensity": st.session_state.get("intensity", 0.8),
                "drone_vol": st.session_state.get("drone_vol", 0.15),
                "seed": st.session_state.get("seed_val", 42)
            }
            st.download_button("Scarica Ora", json.dumps(curr), "preset.json")

    # Pannello Controlli
    st.subheader("🎛️ Controlli Audio-Visivi")
    
    # Carichiamo i valori dal JSON se presente
    ld = {}
    if up_json:
        try: ld = json.load(up_json)
        except: pass

    col1, col2, col3 = st.columns(3)
    with col1:
        v_orig_vol = st.slider("Volume Originale Video", 0.0, 1.0, ld.get("v_orig_vol", 0.3), key="v_orig_vol")
        v_mix = st.slider("Potenza Glitch Generato", 0.0, 5.0, ld.get("v_mix", 1.5), key="v_mix")
    with col2:
        grit = st.slider("Grit (Bitcrush/Distorsione)", 0.0, 1.0, ld.get("grit", 0.7), key="grit")
        g_size = st.slider("Durata Micro-Glitch (s)", 0.01, 0.5, ld.get("g_size", 0.1), key="g_size")
    with col3:
        intensity = st.slider("Sensibilità ai Pixel", 0.0, 2.0, ld.get("intensity", 1.0), key="intensity")
        drone_vol = st.slider("Volume Drone (Texture)", 0.0, 1.0, ld.get("drone_vol", 0.15), key="drone_vol")
        seed = st.number_input("Seed", value=int(ld.get("seed", 42)), key="seed_val")

    # Esecuzione
    if v_file:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        
        if st.button("🚀 GENERA GLITCH ART", use_container_width=True):
            try:
                with st.spinner("Analisi differenze pixel in corso..."):
                    energy = analyze_video_v4(t_v.name)
                    clip = VideoFileClip(t_v.name)
                    
                    params = {
                        "v_orig_vol": v_orig_vol, "v_mix": v_mix, "grit": grit,
                        "g_size": g_size, "intensity": intensity, 
                        "drone_vol": drone_vol, "seed": seed
                    }
                    
                    audio_data = generate_glitch_v4(t_v.name, energy, clip.duration, params)
                    
                    # Export Audio
                    t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(t_a.name, audio_data.T, 44100)
                    
                    # Merge
                    final_clip = clip.set_audio(AudioFileClip(t_a.name))
                    out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_f, codec="libx264", audio_codec="aac")
                    
                    st.success("Sincronizzazione completata!")
                    st.video(out_f)
                    with open(out_f, "rb") as f:
                        st.download_button("💾 Scarica Video Finale", f, "glitch_video.mp4")
                clip.close()
            except Exception as e:
                st.error(f"Errore: {e}")

if __name__ == "__main__":
    main()
