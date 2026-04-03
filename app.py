import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# 1. ANALISI VIDEO
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
    np.random.seed(p["seed"])
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
st.set_page_config(page_title="BeatGlitch V10", layout="wide")
st.title("🌪️ BeatGlitch V10: Audio-Shredder")

presets = {
    "Default (Bilanciato)": {"v_orig": 0.3, "v_mix": 2.5, "stut_ms": 45, "stut_rep": 12, "int": 1.5, "grit": 0.6},
    "Disco Rotto (Locked Groove)": {"v_orig": 0.1, "v_mix": 3.5, "stut_ms": 80, "stut_rep": 25, "int": 2.0, "grit": 0.4},
    "Cyber-Noise (Total)": {"v_orig": 0.0, "v_mix": 4.5, "stut_ms": 15, "stut_rep": 8, "int": 3.5, "grit": 0.95},
    "Ghost (Sussurri)": {"v_orig": 0.05, "v_mix": 1.5, "stut_ms": 120, "stut_rep": 4, "int": 1.0, "grit": 0.2},
    "Radio Interferenza": {"v_orig": 0.2, "v_mix": 3.0, "stut_ms": 5, "stut_rep": 40, "int": 2.5, "grit": 0.98},
    "Glitch-Hop Beats": {"v_orig": 0.4, "v_mix": 3.0, "stut_ms": 30, "stut_rep": 16, "int": 2.2, "grit": 0.5},
    "Deep Drone": {"v_orig": 0.1, "v_mix": 2.0, "stut_ms": 200, "stut_rep": 2, "int": 1.2, "grit": 0.8},
    "Vinyl Scratch": {"v_orig": 0.15, "v_mix": 4.0, "stut_ms": 10, "stut_rep": 30, "int": 3.0, "grit": 0.7}
}

with st.sidebar:
    v_file = st.file_uploader("1. Carica Video", type=["mp4", "mov"])
    a_file = st.file_uploader("2. Audio Esterno (Opzionale)", type=["mp3", "wav"])
    st.markdown("---")
    selected_preset = st.selectbox("🎯 SCEGLI UNO STILE", list(presets.keys()))
    ps = presets[selected_preset]

st.subheader(f"🎛️ Configurazione Attiva: {selected_preset}")
c1, c2, c3 = st.columns(3)
with c1:
    v_orig_vol = st.slider("Volume Originale", 0.0, 1.0, ps["v_orig"])
    v_mix = st.slider("Potenza Glitch", 0.0, 5.0, ps["v_mix"])
with c2:
    stutter_ms = st.slider("Durata Loop (ms)", 5, 250, ps["stut_ms"])
    stutter_reps = st.slider("Ripetizioni Loop", 1, 60, ps["stut_rep"])
with c3:
    intensity = st.slider("Sensibilità Pixel", 0.1, 4.0, ps["int"])
    grit = st.slider("Grit (Sporcizia)", 0.0, 1.0, ps["grit"])

if v_file:
    if st.button("🚀 GENERA REMIX"):
        with st.status("Distruggendo gli atomi sonori...") as s:
            t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_v.write(v_file.read())
            t_a = None
            if a_file:
                t_a = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                t_a.write(a_file.read())
            
            sig = analyze_video_v10(t_v.name)
            clip = VideoFileClip(t_v.name)
            params = {"v_orig_vol":v_orig_vol, "v_mix":v_mix, "intensity":intensity, "stutter_ms":stutter_ms, "stutter_reps":stutter_reps, "grit":grit, "seed":42}
            
            audio = generate_v10_engine(t_v.name, t_a.name if t_a else None, sig, clip.duration, params)
            t_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(t_wav.name, audio.T, 44100)
            
            out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            clip.set_audio(AudioFileClip(t_wav.name)).write_videofile(out, codec="libx264", audio_codec="aac", logger=None)
            st.video(out)
            s.update(label="Remix Pronto!", state="complete")
