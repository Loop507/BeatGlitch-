import streamlit as st
import os
import numpy as np
import tempfile
import traceback
from moviepy.editor import VideoFileClip, AudioFileClip
import librosa
import soundfile as sf
import json

# ──────────────────────────────────────────────────────────────────────────────
#  PRESETS DI FABBRICA (Incluso il nuovo preset per g4.mp4)
# ──────────────────────────────────────────────────────────────────────────────

PRESETS_FACTORY = {
    "📺 Audio-Visual Glitch": {
        "drone_vol": 0.4, "v_mix": 0.8, "auto_m": 0.4,
        "grit": 0.85, "g_size": 0.10, "rnd_f": 0.8, "chaos": 0.7, "intensity": 0.8, "seed": 42
    },
    "🌌 Deep Drone": {
        "drone_vol": 0.8, "v_mix": 0.2, "auto_m": 0.6,
        "grit": 0.1, "g_size": 0.9, "rnd_f": 0.2, "chaos": 0.3, "intensity": 0.5, "seed": 42
    },
    "👾 Hard Glitch": {
        "drone_vol": 0.2, "v_mix": 0.7, "auto_m": 0.1,
        "grit": 0.9, "g_size": 0.1, "rnd_f": 0.8, "chaos": 0.8, "intensity": 0.9, "seed": 666
    },
    "🤖 Cybernetic": {
        "drone_vol": 0.4, "v_mix": 0.5, "auto_m": 0.4,
        "grit": 0.5, "g_size": 0.4, "rnd_f": 0.5, "chaos": 0.5, "intensity": 0.6, "seed": 777
    }
}

# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO ENGINE (Con effetto Stutter e Sincro Avanzato)
# ──────────────────────────────────────────────────────────────────────────────

def generate_ultimate_experimental(video_audio_path, cuts, duration, sr=44100, 
                                  d_vol=0.3, v_mix=0.5, grit=0.6, 
                                  chaos=0.5, g_size=0.4, 
                                  rnd_f=0.5, auto_m=0.2, seed=42):
    
    np.random.seed(int(seed))
    total_samples = int(duration * sr)
    final_output = np.zeros((2, total_samples), dtype=np.float32)
    
    try:
        y_orig, _ = librosa.load(video_audio_path, sr=sr, mono=False)
        if y_orig.ndim == 1: y_orig = np.tile(y_orig, (2, 1))
        if y_orig.shape[1] > total_samples: 
            y_orig = y_orig[:, :total_samples]
        else: 
            y_orig = np.pad(y_orig, ((0,0), (0, total_samples - y_orig.shape[1])))
    except:
        y_orig = np.zeros((2, total_samples), dtype=np.float32)

    # 1. DRONE DI BASE
    t_timeline = np.linspace(0, 1, total_samples, dtype=np.float32)
    freq_drift = 45 + (auto_m * 150 * t_timeline) # Sale di tono verso la fine
    phase = 2 * np.pi * np.cumsum(freq_drift) / sr
    drone = np.sin(phase).astype(np.float32) * d_vol
    final_output += np.tile(drone, (2, 1))

    # 2. GENERAZIONE SUI TAGLI (Glitch & Stutter)
    for i, cut in enumerate(cuts):
        if np.random.random() < (rnd_f * 0.2): continue # Salta casualmente per varietà
            
        start_s = int(cut * sr)
        if start_s >= total_samples: continue

        # --- EFFETTO STUTTER (Ripetizione rapida tipica di g4.mp4) ---
        if np.random.random() < (rnd_f * 0.6):
            # Frammento piccolissimo (da 10 a 30 ms)
            stutter_len_samples = int(sr * np.random.uniform(0.01, 0.03))
            if start_s + stutter_len_samples < total_samples:
                stutter_chunk = y_orig[:, start_s : start_s + stutter_len_samples]
                repeats = np.random.randint(4, 12)
                stutter_wave = np.tile(stutter_chunk, (1, repeats))
                
                s_len = stutter_wave.shape[1]
                end_s = min(start_s + s_len, total_samples)
                actual_s_len = end_s - start_s
                
                # Applichiamo Bitcrush anche allo stutter
                if grit > 0.2:
                    steps = int(2 + (1 - grit) * 10)
                    stutter_wave = np.round(stutter_wave * steps) / steps
                
                final_output[:, start_s:end_s] += stutter_wave[:, :actual_s_len] * v_mix

        # --- EFFETTO GRANULARE STANDARD ---
        eff_len = int(sr * g_size * np.random.uniform(0.5, 1.5))
        end_s = min(start_s + eff_len, total_samples)
        actual_len = end_s - start_s
        
        chunk = y_orig[:, start_s:end_s].copy()
        
        if np.random.random() < (rnd_f * chaos):
            chunk = np.flip(chunk, axis=1) # Reverse
            
        if grit > 0.1:
            steps = int(2 + (1 - grit) * 12)
            chunk = np.round(chunk * steps) / steps
            
        env = np.exp(-5 * np.linspace(0, 1, actual_len, dtype=np.float32))
        final_output[:, start_s:end_s] += chunk * env * (v_mix * 1.5)

    # 3. BACKGROUND SOFT
    final_output += y_orig * (v_mix * 0.1)
    final_output = np.tanh(final_output * (1.1 + chaos))
    
    max_val = np.max(np.abs(final_output))
    if max_val > 0: final_output /= max_val
    return final_output

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch Studio", layout="wide")

def main():
    st.title("🎬 BeatGlitch Studio")

    # Inizializzazione Session State
    if "drone_vol" not in st.session_state:
        for k, v in PRESETS_FACTORY["📺 Audio-Visual Glitch"].items():
            st.session_state[k] = v

    # SIDEBAR
    with st.sidebar:
        st.header("📁 Sorgenti")
        video_file = st.file_uploader("Carica Video", type=["mp4", "mov", "avi"])
        
        st.markdown("---")
        st.header("🎭 Preset Rapidi")
        for name, params in PRESETS_FACTORY.items():
            if st.button(name, use_container_width=True):
                for k, v in params.items(): st.session_state[k] = v
                st.rerun()

        st.markdown("---")
        st.header("💾 Preset Utente")
        # Download
        current_conf = {k: st.session_state[k] for k in PRESETS_FACTORY["📺 Audio-Visual Glitch"].keys()}
        st.download_button("📥 Scarica Config", json.dumps(current_conf), "preset.json", "application/json", use_container_width=True)
        # Upload
        up_p = st.file_uploader("📤 Carica .json", type="json")
        if up_p:
            if st.button("Applica Caricato"):
                d = json.load(up_p)
                for k, v in d.items(): st.session_state[k] = v
                st.rerun()

    # PANNELLO CONTROLLO
    st.subheader("🎛️ Pannello di Controllo")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.slider("Volume Drone", 0.0, 1.0, key="drone_vol")
        st.slider("Mix Audio Video", 0.0, 1.0, key="v_mix")
        st.slider("Auto-Morphing", 0.0, 1.0, key="auto_m")

    with col2:
        st.slider("Grit (Bitcrush)", 0.0, 1.0, key="grit")
        st.slider("Dimensione Grano", 0.05, 1.5, key="g_size")
        st.slider("Random Factor", 0.0, 1.0, key="rnd_f")

    with col3:
        st.slider("Chaos Factor", 0.0, 1.0, key="chaos")
        st.slider("Intensità Totale", 0.0, 1.0, key="intensity")
        st.number_input("Seed", key="seed")

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        if st.button("🚀 GENERA AUDIO-VISUAL GLITCH", use_container_width=True):
            try:
                with st.spinner("Analisi micro-movimenti in corso..."):
                    clip = VideoFileClip(tfile.name)
                    
                    # --- Rilevamento tagli ad alta sensibilità per g4.mp4 ---
                    fps_check = min(clip.fps, 12) # Aumentiamo il sample rate visivo
                    times = np.arange(0, clip.duration, 1.0 / fps_check)
                    cuts = [0.0]
                    prev_f = None
                    for t in times:
                        f = clip.get_frame(t).mean()
                        if prev_f is not None:
                            # Soglia abbassata a 15 per catturare i glitch luminosi
                            if abs(f - prev_f) > 15 and (t - cuts[-1]) >= 0.12:
                                cuts.append(round(t, 3))
                        prev_f = f
                    
                    audio_data = generate_ultimate_experimental(
                        tfile.name, cuts, clip.duration,
                        d_vol=st.session_state.drone_vol,
                        v_mix=st.session_state.v_mix,
                        auto_m=st.session_state.auto_m,
                        grit=st.session_state.grit,
                        g_size=st.session_state.g_size,
                        rnd_f=st.session_state.rnd_f,
                        chaos=st.session_state.chaos,
                        seed=st.session_state.seed
                    )
                    
                    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(temp_audio.name, audio_data.T, 44100)
                    
                    new_audio_clip = AudioFileClip(temp_audio.name)
                    final_clip = clip.set_audio(new_audio_clip)
                    out_v = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_v, codec="libx264", audio_codec="aac", fps=clip.fps)
                    
                    st.video(out_v)
                    with open(out_v, "rb") as f:
                        st.download_button("💾 Scarica Risultato", f, "glitch_art.mp4")
                clip.close()
            except Exception as e:
                st.error(f"Errore: {e}")

if __name__ == "__main__":
    main()
