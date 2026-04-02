import streamlit as st
import os
import numpy as np
import tempfile
import traceback
from moviepy.editor import VideoFileClip, AudioFileClip
import librosa
import soundfile as sf
import json
import cv2  # Analisi Optical Flow

# ──────────────────────────────────────────────────────────────────────────────
#  PRESETS DI FABBRICA
# ──────────────────────────────────────────────────────────────────────────────

PRESETS_FACTORY = {
    "📺 Audio-Visual Glitch (G4)": {
        "drone_vol": 0.4, "v_mix": 0.85, "auto_m": 0.45,
        "grit": 0.85, "g_size": 0.10, "rnd_f": 0.8, "chaos": 0.7, "intensity": 0.9, "seed": 42
    },
    "👾 Hard Glitch": {
        "drone_vol": 0.2, "v_mix": 0.7, "auto_m": 0.1,
        "grit": 0.9, "g_size": 0.05, "rnd_f": 0.8, "chaos": 0.8, "intensity": 0.9, "seed": 666
    },
    "🌌 Deep Drone": {
        "drone_vol": 0.8, "v_mix": 0.2, "auto_m": 0.6,
        "grit": 0.1, "g_size": 0.9, "rnd_f": 0.2, "chaos": 0.3, "intensity": 0.5, "seed": 111
    }
}

# ──────────────────────────────────────────────────────────────────────────────
#  FUNZIONI TECNICHE ANALISI
# ──────────────────────────────────────────────────────────────────────────────

def analyze_motion_optical_flow(video_path):
    """Analizza il movimento dei pixel frame per frame."""
    cap = cv2.VideoCapture(video_path)
    motion_data = []
    prev_gray = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Ridimensiona per non appesantire la RAM
        frame_small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # Differenza tra i frame (movimento)
            diff = cv2.absdiff(gray, prev_gray)
            motion_data.append(np.mean(diff))
        else:
            motion_data.append(0)
        prev_gray = gray
        
    cap.release()
    motion_array = np.array(motion_data)
    if len(motion_array) > 0 and np.max(motion_array) > 0:
        motion_array = motion_array / np.max(motion_array)
    else:
        motion_array = np.zeros(1)
    return motion_array

# ──────────────────────────────────────────────────────────────────────────────
#  MOTORE AUDIO SPERIMENTALE
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

    # Mappa la curva di movimento sulla durata audio
    motion_map = np.interp(np.linspace(0, 1, total_samples), 
                           np.linspace(0, 1, len(motion_curve)), 
                           motion_curve).astype(np.float32)

    # 1. DRONE REATTIVO
    t = np.linspace(0, duration, total_samples, dtype=np.float32)
    base_freq = 40 + (motion_map * 120 * chaos) + (auto_m * 50)
    phase = 2 * np.pi * np.cumsum(base_freq) / sr
    drone = np.sin(phase) * d_vol * (0.4 + motion_map * 0.6)
    audio_out += np.tile(drone, (2, 1))

    # 2. GLITCH & STUTTER
    step = 0.04
    for t_sec in np.arange(0, duration - step, step):
        idx = int(t_sec * sr)
        m_val = motion_map[idx]
        
        if m_val > (1.1 - intensity) or np.random.random() < (rnd_f * 0.15):
            current_g_len = int(sr * g_size * np.random.uniform(0.5, 1.2))
            if idx + current_g_len > total_samples: continue
            
            chunk = y_orig[:, idx : idx + current_g_len].copy()
            
            if m_val > 0.6 and np.random.random() < rnd_f:
                stutter_piece = chunk[:, :int(sr*0.015)]
                if stutter_piece.shape[1] > 0:
                    chunk = np.tile(stutter_piece, (1, max(2, int(current_g_len/stutter_piece.shape[1]))))
                    chunk = chunk[:, :current_g_len]

            local_grit = grit * (0.5 + m_val * 0.5)
            if local_grit > 0.1:
                lev = int(2 + (1 - local_grit) * 15)
                chunk = np.round(chunk * lev) / lev
            
            if np.random.random() < (chaos * 0.4):
                chunk = np.flip(chunk, axis=1)

            env = np.exp(-7 * np.linspace(0, 1, chunk.shape[1]))
            audio_out[:, idx:idx+chunk.shape[1]] += chunk * env * v_mix * m_val

    audio_out = np.tanh(audio_out * (1.2 + chaos))
    if np.max(np.abs(audio_out)) > 0: audio_out /= np.max(np.abs(audio_out))
    
    return audio_out

# ──────────────────────────────────────────────────────────────────────────────
#  INTERFACCIA STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch Pro", layout="wide")

def main():
    st.title("🎬 BeatGlitch: Optical Flow Sync")
    
    # Inizializzazione Session State
    if "drone_vol" not in st.session_state:
        for k, v in PRESETS_FACTORY["📺 Audio-Visual Glitch (G4)"].items():
            st.session_state[k] = v

    with st.sidebar:
        st.header("📁 File Video")
        v_file = st.file_uploader("Carica MP4/MOV", type=["mp4", "mov"])
        
        st.markdown("---")
        st.header("🎭 Preset Veloci")
        for name, p_vals in PRESETS_FACTORY.items():
            if st.button(name, key=f"btn_{name}", use_container_width=True):
                for k, v in p_vals.items(): st.session_state[k] = v
                st.rerun()

        st.markdown("---")
        st.header("💾 Gestione Preset")
        curr_p = {
            "drone_vol": st.session_state.drone_vol,
            "v_mix": st.session_state.v_mix,
            "auto_m": st.session_state.auto_m,
            "grit": st.session_state.grit,
            "g_size": st.session_state.g_size,
            "rnd_f": st.session_state.rnd_f,
            "chaos": st.session_state.chaos,
            "intensity": st.session_state.intensity,
            "seed": st.session_state.seed
        }
        st.download_button("📥 Scarica JSON", json.dumps(curr_p), "preset.json", use_container_width=True)
        up_json = st.file_uploader("📤 Carica JSON", type="json")
        if up_json:
            if st.button("Applica Preset Esterno"):
                data = json.load(up_json)
                for k, v in data.items(): st.session_state[k] = v
                st.rerun()

    st.subheader("🎛️ Parametri Audio-Reattivi")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.slider("Volume Drone", 0.0, 1.0, key="drone_vol")
        st.slider("Mix Audio Video", 0.0, 1.0, key="v_mix")
        st.slider("Auto-Morphing", 0.0, 1.0, key="auto_m")
    with c2:
        st.slider("Grit (Bitcrush)", 0.0, 1.0, key="grit")
        st.slider("Dimensione Grano", 0.01, 1.0, key="g_size")
        st.slider("Random Factor", 0.0, 1.0, key="rnd_f")
    with c3:
        st.slider("Chaos Factor", 0.0, 1.0, key="chaos")
        st.slider("Intensità Reazione", 0.0, 1.0, key="intensity")
        st.number_input("Seed", key="seed")

    if v_file:
        t_v_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(t_v_path, "wb") as f:
            f.write(v_file.read())
        
        if st.button("🚀 GENERA GLITCH REATTIVO", use_container_width=True):
            try:
                with st.spinner("Analisi Optical Flow..."):
                    m_curve = analyze_motion_optical_flow(t_v_path)
                    clip = VideoFileClip(t_v_path)
                    
                    snd = generate_glitch_engine(
                        t_v_path, m_curve, clip.duration,
                        d_vol=st.session_state.drone_vol,
                        v_mix=st.session_state.v_mix,
                        grit=st.session_state.grit,
                        g_size=st.session_state.g_size,
                        rnd_f=st.session_state.rnd_f,
                        chaos=st.session_state.chaos,
                        auto_m=st.session_state.auto_m,
                        intensity=st.session_state.intensity,
                        seed=st.session_state.seed
                    )
                    
                    t_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(t_a.name, snd.T, 44100)
                    
                    final_clip = clip.set_audio(AudioFileClip(t_a.name))
                    out_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_f, codec="libx264", audio_codec="aac")
                    
                    st.success("Sincronizzazione completata!")
                    st.video(out_f)
                    with open(out_f, "rb") as f_out:
                        st.download_button("💾 Scarica Video Finale", f_out, "glitch_art.mp4")
                clip.close()
            except Exception as e:
                st.error(f"Errore: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
