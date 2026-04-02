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
#  PRESETS DEFINITION
# ──────────────────────────────────────────────────────────────────────────────

PRESETS = {
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
#  AUDIO ENGINE (SPERIMENTALE & ASTRATTO)
# ──────────────────────────────────────────────────────────────────────────────

def generate_ultimate_experimental(video_audio_path, cuts, duration, sr=44100, 
                                  d_vol=0.3, v_mix=0.5, grit=0.6, 
                                  chaos=0.5, g_size=0.4, 
                                  rnd_f=0.5, auto_m=0.2, seed=42):
    
    np.random.seed(int(seed))
    total_samples = int(duration * sr)
    # Usiamo float32 per risparmiare memoria su Streamlit Cloud
    final_output = np.zeros((2, total_samples), dtype=np.float32)
    
    # Caricamento audio originale del video
    try:
        y_orig, _ = librosa.load(video_audio_path, sr=sr, mono=False)
        if y_orig.ndim == 1: y_orig = np.tile(y_orig, (2, 1))
        # Match della durata
        if y_orig.shape[1] > total_samples: 
            y_orig = y_orig[:, :total_samples]
        else: 
            y_orig = np.pad(y_orig, ((0,0), (0, total_samples - y_orig.shape[1])))
    except:
        y_orig = np.zeros((2, total_samples), dtype=np.float32)

    t_timeline = np.linspace(0, 1, total_samples, dtype=np.float32)

    # 1. DRONE CON AUTO-MORPH (Evoluzione temporale)
    freq_drift = 40 + (auto_m * 120 * t_timeline)
    phase = 2 * np.pi * np.cumsum(freq_drift) / sr
    drone = np.sin(phase).astype(np.float32) * d_vol
    final_output += np.tile(drone, (2, 1))

    # 2. GENERAZIONE SUI TAGLI
    for i, cut in enumerate(cuts):
        # Probabilità di saltare il taglio (Random Factor)
        if np.random.random() < (rnd_f * 0.3): continue
            
        start_s = int(cut * sr)
        if start_s >= total_samples: continue

        # Dimensione grano influenzata da Auto-Morph e Chaos
        current_g_size = g_size * (1 + (np.random.random() * auto_m))
        eff_len = int(sr * current_g_size)
        end_s = min(start_s + eff_len, total_samples)
        actual_len = end_s - start_s
        
        # Prendiamo il pezzo di audio originale
        chunk = y_orig[:, start_s:end_s].copy()
        
        # --- EFFETTI GLITCH ---
        # Reverse casuale
        if np.random.random() < (rnd_f * chaos):
            chunk = np.flip(chunk, axis=1)
            
        # Bitcrush (Grit)
        if grit > 0.1:
            steps = int(2 + (1 - grit) * 12)
            chunk = np.round(chunk * steps) / steps
            
        # Pitch Shift "sporco" (Auto-morphing)
        shift = 1 + (auto_m * (i / len(cuts) + 0.1))
        if shift > 1.05 and chunk.shape[1] > 10:
            indices = np.arange(0, chunk.shape[1], shift).astype(int)
            chunk_shifted = chunk[:, indices]
            chunk = np.zeros((2, actual_len), dtype=np.float32)
            chunk[:, :chunk_shifted.shape[1]] = chunk_shifted

        # Inviluppo e Mix
        env = np.exp(-4 * np.linspace(0, 1, actual_len, dtype=np.float32))
        final_output[:, start_s:end_s] += chunk * env * (v_mix * 2.0)

    # 3. BACKGROUND AUDIO ORIGINALE (Leggero)
    final_output += y_orig * (v_mix * 0.2)

    # Distorsione finale (Soft Clipping) per amalgamare
    final_output = np.tanh(final_output * (1.2 + chaos))
    
    # Normalizzazione di sicurezza
    max_val = np.max(np.abs(final_output))
    if max_val > 0: final_output /= max_val

    return final_output

# ──────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def detect_cuts(video_path, min_interval):
    try:
        clip = VideoFileClip(video_path)
        fps = min(clip.fps, 8) # Leggero per Streamlit
        times = np.arange(0, clip.duration, 1.0 / fps)
        cuts = [0.0]
        prev_f = None
        for t in times:
            f = clip.get_frame(t).mean()
            if prev_f is not None:
                if abs(f - prev_f) > 25 and (t - cuts[-1]) >= min_interval:
                    cuts.append(round(t, 3))
            prev_f = f
        clip.close()
        return cuts
    except: return [0.0]

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch Studio", layout="wide")

def main():
    st.title("🎬 BeatGlitch Studio")
    st.markdown("Generatore di colonne sonore sperimentali basate sui tagli video.")

    # Inizializzazione Session State per i Preset
    defaults = {
        "drone_vol": 0.3, "v_mix": 0.5, "auto_m": 0.2, "grit": 0.6,
        "g_size": 0.4, "rnd_f": 0.5, "chaos": 0.5, "intensity": 0.7, "seed": 42
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # SIDEBAR: Caricamento File
    with st.sidebar:
        st.header("📁 Sorgenti")
        video_file = st.file_uploader("Carica Video", type=["mp4", "mov", "avi"])
        
        st.markdown("---")
        st.header("💾 Preset & Config")
        # Pulsanti Preset
        for name, params in PRESETS.items():
            if st.button(name, use_container_width=True):
                for k, v in params.items(): st.session_state[k] = v
                st.rerun()

    # PANNELLO CENTRALE: Parametri sempre visibili
    st.subheader("🎛️ Pannello di Controllo Sound Design")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌑 Base & Drone**")
        st.session_state.drone_vol = st.slider("Volume Drone", 0.0, 1.0, key="drone_vol_s", value=st.session_state.drone_vol)
        st.session_state.v_mix = st.slider("Mix Audio Video", 0.0, 1.0, key="v_mix_s", value=st.session_state.v_mix)
        st.session_state.auto_m = st.slider("Auto-Morphing", 0.0, 1.0, key="auto_m_s", value=st.session_state.auto_m)

    with col2:
        st.markdown("**👾 Textures & Glitch**")
        st.session_state.grit = st.slider("Grit (Bitcrush)", 0.0, 1.0, key="grit_s", value=st.session_state.grit)
        st.session_state.g_size = st.slider("Dimensione Grano", 0.05, 1.5, key="g_size_s", value=st.session_state.g_size)
        st.session_state.rnd_f = st.slider("Random Factor", 0.0, 1.0, key="rnd_f_s", value=st.session_state.rnd_f)

    with col3:
        st.markdown("**☣️ Evoluzione**")
        st.session_state.chaos = st.slider("Chaos Factor", 0.0, 1.0, key="chaos_s", value=st.session_state.chaos)
        st.session_state.intensity = st.slider("Intensità Totale", 0.0, 1.0, key="intensity_s", value=st.session_state.intensity)
        st.session_state.seed = st.number_input("Seed (Riproducibilità)", key="seed_n", value=int(st.session_state.seed))

    # ESECUZIONE
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        if st.button("🚀 GENERA SOUND DESIGN", use_container_width=True):
            try:
                with st.spinner("Analisi video e generazione audio..."):
                    cuts = detect_cuts(tfile.name, 0.3)
                    clip = VideoFileClip(tfile.name)
                    duration = clip.duration
                    
                    # Genera l'audio sperimentale
                    audio_data = generate_ultimate_experimental(
                        tfile.name, cuts, duration,
                        d_vol=st.session_state.drone_vol,
                        v_mix=st.session_state.v_mix,
                        auto_m=st.session_state.auto_m,
                        grit=st.session_state.grit,
                        g_size=st.session_state.g_size,
                        rnd_f=st.session_state.rnd_f,
                        chaos=st.session_state.chaos,
                        seed=st.session_state.seed
                    )
                    
                    # Salvataggio temporaneo audio
                    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(temp_audio.name, audio_data.T, 44100)
                    
                    # Unione Video + Nuovo Audio
                    new_audio_clip = AudioFileClip(temp_audio.name)
                    final_clip = clip.set_audio(new_audio_clip)
                    
                    out_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    final_clip.write_videofile(out_video, codec="libx264", audio_codec="aac")
                    
                    st.success("✅ Generazione completata!")
                    st.video(out_video)
                    
                    with open(out_video, "rb") as f:
                        st.download_button("💾 Scarica Video", f, "beatglitch_output.mp4")
                
                clip.close()
                new_audio_clip.close()
            except Exception as e:
                st.error(f"Errore: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()  
