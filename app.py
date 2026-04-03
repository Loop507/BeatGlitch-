# BeatGlitch – MP3/WAV Toggle (Streamlit Cloud Ready)

import os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment

MAX_DURATION = 30
PREVIEW_SR = 22050
FINAL_SR = 44100
FRAME_ANALYSIS_FPS = 10
CHUNK_SEC = 5

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

        energy.append(0.6 * motion + 0.4 * lum)
        prev_gray = gray
        i += 1

    cap.release()

    energy = np.array(energy, dtype=np.float32)
    if energy.max() > 0:
        energy /= energy.max()

    return energy


def limiter(x):
    return np.tanh(x)


def generate_audio(video_path, energy, duration, sr, params):
    N = int(duration * sr)
    chunk_size = int(CHUNK_SEC * sr)

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


st.title("BeatGlitch – MP3/WAV Edition")

mode = st.radio("Mode", ["Preview", "Final"])
format_choice = st.radio("Audio Format", ["WAV", "MP3"])

file = st.file_uploader("Upload video", type=["mp4","mov"])

if file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file.read())
    path = tmp.name

    clip = VideoFileClip(path)
    duration = clip.duration

    if duration > MAX_DURATION:
        st.error("Max 30s allowed")
        st.stop()

    energy = analyze_video(path)
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

        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        if format_choice == "WAV":
            out_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(out_audio.name, audio.T, sr)
            st.audio(out_audio.name)

        else:
            audio_segment = AudioSegment(
                audio_int16.T.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=2
            )

            out_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            audio_segment.export(out_audio.name, format="mp3", bitrate="192k")
            st.audio(out_audio.name)

        if mode == "Final":
            out_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            clip.set_audio(AudioFileClip(out_audio.name)).write_videofile(
                out_video,
                codec="libx264",
                preset="ultrafast",
                audio_codec="aac",
                logger=None
            )

            st.video(out_video)

            with open(out_video, "rb") as f:
                st.download_button("Download video", f, "glitch.mp4")
