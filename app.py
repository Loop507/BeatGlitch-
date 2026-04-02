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
    "🔥 Hyper-Reactive (G4)": {
        "drone_vol": 0.2, "v_mix": 1.0, "auto_m": 0.5,
        "grit": 0.9, "g_size": 0.05, "rnd_f": 0.9, "chaos": 0.9, "intensity": 1.0, "seed": 42
    },
    "👾 Hard Glitch": {
        "drone_vol": 0.2, "v_mix": 0.7, "auto_m": 0.1,
        "grit": 0.9, "g_size": 0.05, "rnd_f": 0.8, "chaos": 0.8, "intensity": 0.9, "seed": 666
    },
    "🌌 Deep Drone": {
        "drone_vol": 0.8, "v_mix": 0.3, "auto_m": 0.7,
        "grit": 0.2, "g_size": 0.8, "rnd_f": 0.3, "chaos": 0.4, "intensity": 0.6, "seed": 111
    }
}

# ──────────────────────────────────────────────────────────────────────────────
#  ANALISI MOVIMENTO (OPTICAL FLOW)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_motion_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    motion_data = []
    prev_gray = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        # Analisi su scala ridotta per velocità
        frame_small = cv2.resize(frame, (120, 90))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_data.append(np.mean(diff))
        else:
            motion_data.append(0)
        prev_gray = gray
    cap.release()
    
    motion_array = np.array(motion_data)
    if len(motion_array) > 0 and np.max(motion_array) > 0:
        # Eleviamo al quadrato per enfatizzare solo i movimenti bruschi
        motion_array = (motion_array / np.max(motion_array)) ** 2
    else:
        motion_array = np.zeros(1)
    return motion_array

# ──────────────────────────────────────────────────────────────────────────────
#  MOTORE AUDIO
# ──────────────────────────────────────────────────────────────────────────────

def generate_glitch_engine(video_path, motion_curve, duration, params, sr=44100):
    np.random.seed(int(params['seed']))
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

    # 1. Drone Texture
    t = np.linspace(0, duration, total_samples, dtype=np.float32)
    freq = 40 + (params['auto_m'] * 60)
    drone = np.sin(2 * np.pi * freq * t) * params['drone_vol'] * (1.0 - motion_map * 0.4)
    audio_out += np.tile(drone, (2, 1))

    # 2. Glitch Engine (Step 0.01 per micro-sincronia)
    step = 0.01 
    for t_sec in np.arange(0, duration - 0.1, step):
        idx = int(t_sec * sr)
        m_val = motion_map[idx]
        
        if m_val > (0.6 * (1.1 - params['intensity'])) or np.random.random() < (params['rnd_f'] * 0.05):
            g_dur = np.random.uniform(0.01, params['g_size'])
            g_len = int(sr * g_dur)
            if idx + g_len > total_samples: continue
            
            chunk = y_orig[:, idx : idx + g_len].copy()
            
            # Stutter su picchi di movimento
            if m_val > 0.4:
                sub = int(sr * 0.006)
                if sub < chunk.shape[1]:
                    chunk = np.tile(chunk[:, :sub], (1, int(g_len/sub) + 1))[:, :g_len]

            # Bitcrush
            lev = int(2 + (1 - params['grit']) * 10)
            chunk = np.round(chunk * lev) / lev
            
            if np.random.random() < (params['chaos'] * 0.3):
                chunk = np.flip(chunk, axis=1)

            env = np.hanning(chunk.shape[1])
            audio_out[:, idx:idx+chunk.shape[1]] += chunk * env * params['v_mix'] * m_val

    audio_out = np.clip(audio_out * 1.5, -1, 1)
    return audio_out

# ──────────────────────────────────────────────────────────────────────────────
#  INTERFACCIA
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch Studio Pro", layout="wide")

def main():
    st.title("🎬 BeatGlitch: Hyper-Sync Studio")

    # Inizializzazione Session State
    if "drone_vol" not in st.session_state:
        for k, v in PRESETS_FACTORY["🔥 Hyper-Reactive (G4)"].items():
            st.session_state[k] = v

    with st.sidebar:
        st.header("📁 File & Preset")
        v_file = st.file_uploader("Carica Video", type=["mp4", "mov"])
        
        st.markdown("---")
        for name, p_vals in PRESETS_FACTORY.items():
            if st.button(name, use_container_width=True):
                for k, v in p_vals.items(): st.session_state[k] = v
                st.rerun()

        st.markdown("---")
        st.header("💾 Salva/Carica")
        curr_p = {k: st.session_state[k] for k in PRESETS_FACTORY["🔥 Hyper-Reactive (G4)"].keys()}
        st.download_button("📥 Scarica Preset JSON", json.dumps(curr_p), "glitch_preset.json", use_container_width=True)
        
        up_json = st.file_uploader("📤 Carica JSON", type="json")
        if up_json:
            if st.button("Applica Caricato"):
                d = json.load(up_json)
                for k, v in d.items(): st.session_state[k] = v
                st.rerun()

    st.subheader("🎛️ Pannello di Controllo")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.slider("Volume Drone", 0.0, 1.0, key="drone_vol")
        st.slider("Mix Audio Video", 0.0, 2.0, key="v_mix")
        st.slider("Auto-Morphing", 0.0, 1.0, key="auto_m")
    with c2:
        st.slider("Grit (Distorsione)", 0.0, 1.0, key="grit")
        st.slider("Dimensione Grano", 0.01, 0.5, key="g_size")
        st.slider("Random Factor", 0.0, 1.0, key="rnd_f")
    with c3:
        st.slider("Chaos Factor", 0.0, 1.0, key="chaos")
        st.slider("Intensità Reazione", 0.0, 1.0, key="intensity")
        st.number_input("Seed", key="seed")

    if v_file:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read())
        
        if st.button("🚀 GENERA GLITCH ART SINCRO", use_container_width=True):
            try:
                with st.spinner("Analisi sub-frame e Optical Flow..."):
                    m_curve = analyze_motion_optical_flow(t_v.name)
                    clip = VideoFileClip(t_v.name)
                    
                    # Raccogliamo i parametri correnti
                    p_dict = {k: st.session_state[k] for k in ["drone_vol","v_mix","auto_m","grit","g_size","rnd_f","chaos","intensity","seed"]}
                    
                    snd = generate_glitch_engine(t_v.name, m_curve, clip.duration, p_dict)
                    
                    t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(t_a.name, snd.T, 44100)
                    
                    final_clip = clip.set_audio(AudioFileClip(t_a.name))
                    out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_f, codec="libx264")
                    
                    st.video(out_f)
                    st.download_button("💾 Scarica Risultato", open(out_f, "rb"), "glitch_art.mp4")
                clip.close()
            except Exception as e:
                st.error(f"Errore: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
