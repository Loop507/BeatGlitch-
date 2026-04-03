# BeatGlitch – Cloud Optimized Version (Streamlit Community Cloud)
# Focus: low RAM, chunk processing, preview mode, caching

import os, tempfile, json, time
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_DURATION = 30
PREVIEW_SR = 22050
FINAL_SR = 44100
FRAME_ANALYSIS_FPS = 10
CHUNK_SEC = 5

# ─────────────────────────────────────────────
# VIDEO ANALYSIS (CACHED + DOWNSAMPLED)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = max(1, int(fps / FRAME_ANALYSIS_FPS))

    energy = []
    prev_gray = None

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if i % frame_skip != 0:
            i += 1
            continue

        small = cv2.resize(frame, (120, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        lum = np.mean(gray) / 255.0

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = np.mean(diff) / 255.0
        else:
            motion = 0

        val = (0.6 * motion + 0.4 * lum)
        energy.append(val)

        prev_gray = gray
        i += 1

    cap.release()

    energy = np.array(energy, dtype=np.float32)
    if energy.max() > 0:
        energy /= energy.max()

    return energy, fps

# ─────────────────────────────────────────────
# AUDIO ENGINE (CHUNKED)
# ─────────────────────────────────────────────

def limiter(x):
    return np.tanh(x)


def generate_audio(video_path, energy, duration, sr, params):
    N = int(duration * sr)
    chunk_size = int(CHUNK_SEC * sr)

    # Load source
    try:
        src, _ = librosa.load(video_path, sr=sr, mono=False)
        if src.ndim == 1:
            src = np.tile(src, (2,1))
    except:
        src = np.random.randn(2, N).astype(np.float32) * 0.3

    if src.shape[1] < N:
        src = np.pad(src, ((0,0),(0, N - src.shape[1])))
    else:
        src = src[:, :N]

    out = np.zeros((2, N), dtype=np.float32)

    e_map = np.interp(
        np.linspace(0, 1, N),
        np.linspace(0, 1, len(energy)),
        energy
    )

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk_len = end - i

        e = e_map[i:end]
        noise = np.random.randn(2, chunk_len).astype(np.float32)

        out[:, i:end] += noise * e * params["intensity"]

    drone = np.sin(2*np.pi*55*np.linspace(0,duration,N)) * params["drone"]
    out += np.tile(drone, (2,1))

    mix = src * params["orig"] + out * params["mix"]

    return limiter(mix)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.set_page_config(layout="wide")
st.title("BeatGlitch – Cloud Edition")

mode = st.radio("Mode", ["Preview", "Final"])

file = st.file_uploader("Upload video", type=["mp4","mov"])

if file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file.read())
    path = tmp.name

    clip = VideoFileClip(path)
    duration = clip.duration

    if duration > MAX_DURATION:
        st.error("Max 30s allowed on cloud")
        st.stop()

    energy, fps = analyze_video(path)

    st.line_chart(energy)

    params = {
        "orig": st.slider("Original",0.0,1.0,0.3),
        "mix": st.slider("Glitch",0.0,3.0,1.5),
        "intensity": st.slider("Intensity",0.0,2.0,1.0),
        "drone": st.slider("Drone",0.0,1.0,0.1)
    }

    if st.button("Generate"):
        sr = PREVIEW_SR if mode == "Preview" else FINAL_SR

        audio = generate_audio(path, energy, duration, sr, params)

        st.audio(audio.T, sample_rate=sr)

        if mode == "Final":
            wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(wav.name, audio.T, sr)

            out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            clip.set_audio(AudioFileClip(wav.name)).write_videofile(
                out, codec="libx264", preset="ultrafast", audio_codec="aac", logger=None
            )

            st.video(out)

            with open(out, "rb") as f:
                st.download_button("Download", f, "glitch.mp4")

            os.unlink(wav.name)
