import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# 1. ANALISI SENSORI VIDEO
def analyze_video_v10(video_path):
    cap = cv2.VideoCapture(video_path)
    sig = {"lum": [], "mot": [], "hue": [], "var": []}
    prev_gray = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        img = cv2.resize(frame, (100, 75))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sig["lum"].append(np.mean(gray) / 255.0)
        sig["hue"].append(np.mean(hsv[:,:,0]) / 180.0)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray); m = np.mean(diff) / 255.0
            sig["mot"].append(m)
            sig["var"].append(abs(m - (sig["mot"][-2] if len(sig["mot"])>1 else 0)))
        else:
            sig["mot"].append(0.0); sig["var"].append(0.0)
        prev_gray = gray
    cap.release()
    for k in sig:
        arr = np.array(sig[k])
        sig[k] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return sig

# 2. MOTORE DI DISTRUZIONE
def generate_v10_engine(video_path, audio_ext_path, sig, duration, p, sr=44100):
    np.random.seed(int(p["seed"])) # Il Seed blocca la casualità della tempesta
    N = int(duration * sr)
    t_ax = np.linspace(0, duration, N)
    mot = np.interp(t_ax, np.linspace(0, duration, len(sig["mot"])), sig["mot"])
    var = np.interp(t_ax, np.linspace(0, duration, len(sig["var"])), sig["var"])
    lum = np.interp(t_ax, np.linspace(0, duration, len(sig["lum"])), sig["lum"])
    hue = np.interp(t_ax, np.linspace(0, duration, len(sig["hue"])), sig["hue"])

    try:
        path = audio_ext_path if audio_ext_path else video_path
        y_src, _ = librosa.load(path, sr=sr, mono=True)
        if len(y_src) > N: y_src = y_src[:N]
        else: y_src = np.tile(y_src, int(np.ceil(N/len(y_src))))[:N]
    except:
        y_src = np.random.uniform(-0.3, 0.3, N) * lum

    out_audio = np.zeros(N)
    s_len = int(sr * (p["stutter_ms"] / 1000.0))
    idx = 0
    while idx < N - max(s_len * p["stutter_reps"], 2000):
        if var[idx] * p["intensity"] > 0.8:
            frag = y_src[idx : idx + s_len].copy()
            res = max(2, int(2 + (1-p["grit"])*16))
            frag = np.round(frag * res) / res
            for r in range(p["stutter_reps"]):
                pos = idx + (r * s_len)
                if pos + s_len < N: out_audio[pos : pos + s_len] += frag * np.hanning(s_len) * p["v_mix"]
            idx += s_len * p["stutter_reps"]
        elif (mot[idx] * p["intensity"]) > 0.1:
            g_len = int(sr * np.random.uniform(0.005, 0.04))
            grain = y_src[idx : idx + g_len].copy()
            res = max(2, int(2 + (1-p["grit"]) * 20 * (1-hue[idx])))
            grain = np.round(grain * res) / res
            out_audio[idx : idx + g_len] += grain * np.hanning(g_len) * p["v_mix"] * (mot[idx] * p["intensity"])
            idx += int(sr * 0.002)
        else: idx += int(sr * 0.005)

    final = (y_src * p["v_orig_vol"]) + (out_audio * 0.8)
    return np.tile(np.clip(final, -1.0, 1.0), (2, 1))

# 3. INTERFACCIA
st.set_page_config(page_title="BeatGlitch V10 Pro", layout="wide")
st.title("🌪️ BeatGlitch V10 Pro: Preset Manager")

# Default Presets
presets_lib = {
    "Default (Bilanciato)": {"v_orig": 0.3, "v_mix": 2.5, "stut_ms": 45, "stut_rep": 12, "int": 1.5, "grit": 0.6, "seed": 42},
    "Disco Rotto": {"v_orig": 0.1, "v_mix": 3.5, "stut_ms": 80, "stut_rep": 25, "int": 2.0, "grit": 0.4, "seed": 77},
    "Cyber-Noise": {"v_orig": 0.0, "v_mix": 4.5, "stut_ms": 15, "stut_rep": 8, "int": 3.5, "grit": 0.95, "seed": 666}
}

with st.sidebar:
    st.header("📂 File & Preset")
    v_file = st.file_uploader("Video", type=["mp4", "mov"])
    a_file = st.file_uploader("Audio Esterno", type=["mp3", "wav"])
    
    st.markdown("---")
    # Caricamento Preset Esterno
    preset_upload = st.file_uploader("Carica Preset JSON", type="json")
    
    # Logica di caricamento valori
    if preset_upload:
        config = json.load(preset_upload)
    else:
        sel = st.selectbox("🎯 Preset Rapidi", list(presets_lib.keys()))
        config = presets_lib[sel]

st.subheader("🎛️ Pannello di Controllo")
c1, c2, c3 = st.columns(3)
with c1:
    v_orig_vol = st.slider("Volume Originale", 0.0, 1.0, config.get("v_orig", 0.3))
    v_mix = st.slider("Potenza Glitch", 0.0, 5.0, config.get("v_mix", 2.5))
    seed_val = st.number_input("🎲 Seed (Identità)", value=config.get("seed", 42))

with c2:
    stutter_ms = st.slider("Loop ms", 5, 250, config.get("stut_ms", 45))
    stutter_reps = st.slider("Ripetizioni", 1, 60, config.get("stut_rep", 12))

with c3:
    intensity = st.slider("Sensibilità", 0.1, 4.0, config.get("int", 1.5))
    grit = st.slider("Grit", 0.0, 1.0, config.get("grit", 0.6))

# Tasto per scaricare il Preset attuale
current_params = {
    "v_orig": v_orig_vol, "v_mix": v_mix, "stut_ms": stutter_ms, 
    "stut_rep": stutter_reps, "int": intensity, "grit": grit, "seed": seed_val
}
st.sidebar.download_button(
    label="💾 Salva Preset Attuale",
    data=json.dumps(current_params),
    file_name="mio_preset_glitch.json",
    mime="application/json"
)

if v_file:
    if st.button("🚀 GENERA", use_container_width=True):
        with st.status("Elaborazione...") as s:
            t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_v.write(v_file.read())
            t_a_ext = None
            if a_file:
                t_a_ext = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                t_a_ext.write(a_file.read())
            
            sig = analyze_video_v10(t_v.name)
            clip = VideoFileClip(t_v.name)
            
            audio = generate_v10_engine(t_v.name, t_a_ext.name if t_a_ext else None, sig, clip.duration, current_params)
            t_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(t_wav.name, audio.T, 44100)
            
            out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            clip.set_audio(AudioFileClip(t_wav.name)).write_videofile(out, codec="libx264", audio_codec="aac", logger=None)
            st.video(out)
            s.update(label="Completato!", state="complete")
