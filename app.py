import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# --- 1. ANALISI SENSORI VIDEO ---
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
            sig["var"].append(abs(m - (sig["mot"][-2] if len(sig["mot"]) > 1 else 0)))
        else:
            sig["mot"].append(0.0); sig["var"].append(0.0)
        prev_gray = gray
    cap.release()
    for k in sig:
        arr = np.array(sig[k])
        sig[k] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return sig

# --- 2. MOTORE DI DISTRUZIONE (CON DRONE E SINTESI) ---
def generate_v10_engine(video_path, audio_ext_path, sig, duration, p, sr=44100):
    np.random.seed(int(p["seed"]))
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
        if np.max(np.abs(y_src)) < 1e-3: raise ValueError("Vuoto")
    except:
        y_src = np.random.uniform(-0.2, 0.2, N) * lum

    out_glitch = np.zeros(N)
    s_len = int(sr * (p["stutter_ms"] / 1000.0))
    idx = 0
    while idx < N - max(s_len * p["stutter_reps"], 2000):
        if var[idx] * p["intensity"] > 0.8:
            frag = y_src[idx : idx + s_len].copy()
            res = max(2, int(2 + (1-p["grit"])*16))
            frag = np.round(frag * res) / res
            for r in range(p["stutter_reps"]):
                pos = idx + (r * s_len)
                if pos + s_len < N: out_glitch[pos : pos + s_len] += frag * np.hanning(s_len) * p["v_mix"]
            idx += s_len * p["stutter_reps"]
        elif (mot[idx]
