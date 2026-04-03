import time, json, os, tempfile
import numpy as np
import cv2
import librosa
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip

# ─────────────────────────────────────────────────────────────────
# 1. ANALISI VIDEO  →  energy[], scene_cuts[]
# ─────────────────────────────────────────────────────────────────

def analyze_video(video_path, res=(160, 120)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    lum_s, det_s, mot_s, var_s, hue_s, hist_s = [], [], [], [], [], []
    prev_gray = prev_hue = prev_hist = None
    prev_motion = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame, res)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lum_s.append(float(np.mean(gray)) / 255.0)
        det_s.append(float(np.std(gray)) / 255.0)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mot  = float(np.mean(diff)) / 255.0
        else: mot = 0.0
        mot_s.append(mot)
        var_s.append(abs(mot - prev_motion))
        prev_motion = mot
        prev_gray = gray
        hue = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.float32)
        if prev_hue is not None:
            d = np.abs(hue - prev_hue)
            d = np.minimum(d, 180 - d)
            hue_s.append(float(np.mean(d) + np.std(d)))
        else: hue_s.append(0.0)
        prev_hue = hue
        h = cv2.calcHist([small],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]).flatten()
        h /= h.sum() + 1e-6
        if prev_hist is not None: hist_s.append(float(np.sum(np.abs(h - prev_hist))))
        else: hist_s.append(0.0)
        prev_hist = h
    cap.release()

    def norm(x):
        a = np.array(x, dtype=np.float32)
        mx = a.max()
        return a / mx if mx > 0 else np.full_like(a, 0.1)

    sig = {
        "lum": norm(lum_s), "detail": norm(det_s), "motion": norm(mot_s),
        "var_motion": norm(var_s), "hue": norm(hue_s), "scene": norm(hist_s),
    }
    # Calcolo energia composita
    energy = (0.30 * sig["motion"] + 0.25 * sig["var_motion"] + 0.20 * sig["hue"] + 0.15 * sig["detail"] + 0.10 * sig["scene"])
    mx = energy.max()
    if mx > 0: energy /= mx
    sig["energy"] = np.power(energy, 1.2)
    scene_cuts = np.where(sig["scene"] > np.percentile(sig["scene"], 90))[0]
    return sig, scene_cuts, fps

# ─────────────────────────────────────────────────────────────────
# 2. MOTORE AUDIO – SINCRO TOTALE
# ─────────────────────────────────────────────────────────────────

def generate_glitch_audio(video_path, sig_dict, scene_cuts, fps, duration, p, sr=44100):
    np.random.seed(int(p["seed"]))
    N = int(duration * sr)
    energy = sig_dict["energy"] # <--- FIX: estraiamo l'energy dal dizionario

    src = None
    try:
        raw, _ = librosa.load(video_path, sr=sr, mono=False)
        if raw.ndim == 1: raw = np.tile(raw, (2, 1))
        if np.sqrt(np.mean(raw**2)) > 1e-4: src = raw
    except: pass

    if src is None:
        rng = np.random.default_rng(int(p["seed"]))
        noise = rng.standard_normal((2, N)).astype(np.float32)
        src = noise * 0.4 

    if src.shape[1] < N: src = np.pad(src, ((0,0),(0, N - src.shape[1])))
    else: src = src[:, :N]

    out = np.zeros((2, N), dtype=np.float32)
    frame_times = np.linspace(0, duration, len(energy))

    # MECCANISMO 1 – MICRO-REAZIONE
    g_max = int(sr * p["g_size"])
    g_min = int(sr * 0.002)
    active_frames = np.where(energy > 0.05)[0]

    for fi in active_frames:
        t = frame_times[fi]; i = int(t * sr)
        pwr = energy[fi] * p["intensity"]
        n_g = max(1, int(pwr * 8))
        for _ in range(n_g):
            g_len = np.random.randint(g_min, max(g_min+1, int(g_max * pwr)))
            jitter = np.random.randint(-int(sr*0.01), int(sr*0.01)+1)
            start = int(np.clip(i + jitter, 0, N - g_len - 1))
            grain = src[:, start:start+g_len].copy()
            if p["grit"] > 0:
                steps = max(2, int(2 + (1.0 - p["grit"]) * 16))
                grain = np.round(grain * steps) / steps
            env = np.hanning(g_len).astype(np.float32)
            out[:, start:start+g_len] += grain * env * p["v_mix"] * (pwr + 0.05)

    # MECCANISMO 2 – STUTTER
    stutter_len = int(sr * p["stutter_ms"] / 1000.0)
    stutter_reps = int(p["stutter_reps"])
    peak_thresh = float(np.percentile(energy, 85))
    peak_frames = set(np.where(energy > peak_thresh)[0].tolist())
    peak_frames.update(scene_cuts.tolist())

    written_stutter = set()
    for fi in sorted(peak_frames):
        t = frame_times[fi]; i = int(t * sr)
        if i + stutter_reps * stutter_len >= N: continue
        if any(abs(fi - w) < 5 for w in written_stutter): continue
        written_stutter.add(fi)
        chunk = src[:, i:i+stutter_len].copy()
        for rep in range(stutter_reps):
            s = i + rep * stutter_len
            env = np.hanning(stutter_len).astype(np.float32)
            decay = (1.0 - rep / stutter_reps) ** 1.5
            out[:, s:s+stutter_len] += chunk * env * p["v_mix"] * energy[fi] * decay

    # MECCANISMO 3 – DRONE
    t_ax = np.linspace(0, duration, N, dtype=np.float32)
    e_map = np.interp(t_ax, frame_times, energy)
    drone = np.sin(2 * np.pi * 55 * t_ax) * p["drone_vol"] * (0.1 + e_map * 0.9)
    out += np.tile(drone, (2, 1))

    mix = (src * p["v_orig_vol"]) + (out * 0.7)
    return np.clip(mix, -1.0, 1.0)

# ─────────────────────────────────────────────────────────────────
# 3. INTERFACCIA STREAMLIT
# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch – Sincro Totale", layout="wide")
st.title("🎬 BeatGlitch – Sincro Totale")

# Gestione Preset Caricati (Inizializzazione session_state)
if "v_orig_vol" not in st.session_state:
    st.session_state.update({"v_orig_vol":0.25, "v_mix":2.0, "grit":0.8, "g_size":0.04, "intensity":1.3, "drone_vol":0.12, "stutter_ms":80, "stutter_reps":6, "seed":42})

with st.sidebar:
    st.header("📂 Risorse")
    v_file = st.file_uploader("Carica Video", type=["mp4","mov"])
    up_json = st.file_uploader("Carica Preset JSON", type=["json"])
    
    if up_json:
        try:
            ld = json.load(up_json)
            for k, v in ld.items(): st.session_state[k] = v
            st.success("Preset Caricato!")
        except: st.error("JSON non valido")

    if st.button("📥 Prepara Download Preset"):
        curr = {k: st.session_state[k] for k in ["v_orig_vol","v_mix","grit","g_size","intensity","drone_vol","stutter_ms","stutter_reps","seed"]}
        st.download_button("Scarica Ora", json.dumps(curr, indent=2), "preset.json")

# UI Slider usando session_state
c1, c2, c3 = st.columns(3)
with c1:
    v_orig_vol = st.slider("Volume Originale", 0.0, 1.0, key="v_orig_vol")
    v_mix      = st.slider("Potenza Glitch",   0.0, 5.0, key="v_mix")
with c2:
    grit       = st.slider("Grit (Bitcrush)",  0.0, 1.0, key="grit")
    g_size     = st.slider("Durata Grano (s)", 0.002, 0.1, key="g_size")
with c3:
    intensity  = st.slider("Intensità Reazione", 0.0, 2.0, key="intensity")
    drone_vol  = st.slider("Drone 55 Hz",      0.0, 1.0, key="drone_vol")

st.subheader("🔁 Stutter")
cs1, cs2, cs3 = st.columns(3)
with cs1: stutter_ms = st.slider("Durata Loop (ms)", 10, 300, key="stutter_ms")
with cs2: stutter_reps = st.slider("Ripetizioni Stutter", 1, 16, key="stutter_reps")
with cs3: seed = st.number_input("Seed", key="seed")

if v_file:
    t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t_v.write(v_file.read()); t_v.flush()
    
    if st.button("🚀 GENERA GLITCH ART", use_container_width=True):
        p = { "v_orig_vol": v_orig_vol, "v_mix": v_mix, "grit": grit, "g_size": g_size, "intensity": intensity, "drone_vol": drone_vol, "stutter_ms": stutter_ms, "stutter_reps": stutter_reps, "seed": seed }
        try:
            with st.status("Elaborazione in corso...") as status:
                sig, cuts, fps = analyze_video(t_v.name)
                clip = VideoFileClip(t_v.name)
                audio = generate_glitch_audio(t_v.name, sig, cuts, fps, clip.duration, p)
                
                wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(wav_tmp.name, audio.T, 44100)
                
                out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                clip.set_audio(AudioFileClip(wav_tmp.name)).write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
                
                st.video(out_path)
                with open(out_path,"rb") as f:
                    st.download_button("💾 Scarica Risultato", f, "glitch_video.mp4")
                status.update(label="✅ Completato!", state="complete")
        except Exception as e: st.error(f"Errore: {e}")
