"""
BeatGlitch – Sincro Totale
Tre meccanismi:
  1. Micro-reazione    – ogni frame con energia > soglia genera un burst granulare
  2. Stutter           – quando c'è un picco, un loop breve del suono si ripete N volte
  3. Sincro totale     – i grani sono estratti ESATTAMENTE dal momento temporale
                         corrispondente al frame sorgente, non da posizioni casuali
"""

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
    motion_s, color_s, hist_s = [], [], []
    prev_gray = prev_hue = prev_hist = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, res)

        # A) Movimento luminanza
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_s.append(float(np.mean(diff) + np.std(diff)))
        else:
            motion_s.append(0.0)
        prev_gray = gray

        # B) Shift colore (Hue circolare)
        hue = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.float32)
        if prev_hue is not None:
            d = np.abs(hue - prev_hue)
            d = np.minimum(d, 180 - d)
            color_s.append(float(np.mean(d) + np.std(d)))
        else:
            color_s.append(0.0)
        prev_hue = hue

        # C) Distanza istogramma (scene cut)
        hist = cv2.calcHist([small], [0,1,2], None, [8,8,8],
                            [0,256,0,256,0,256]).flatten()
        hist /= hist.sum() + 1e-6
        if prev_hist is not None:
            hist_s.append(float(np.sum(np.abs(hist - prev_hist))))
        else:
            hist_s.append(0.0)
        prev_hist = hist

    cap.release()

    def norm(x):
        a = np.array(x, dtype=np.float32)
        mx = a.max()
        return a / mx if mx > 0 else np.full_like(a, 0.1)

    energy = 0.40 * norm(motion_s) + 0.35 * norm(color_s) + 0.25 * norm(hist_s)
    mx = energy.max()
    if mx > 0:
        energy /= mx
    energy = np.power(energy, 1.3)

    # Scene cuts: frame dove l'istogramma salta sopra soglia adattiva
    scene_arr = norm(hist_s)
    threshold = float(np.percentile(scene_arr, 90))
    scene_cuts = np.where(scene_arr > threshold)[0]

    return energy, scene_cuts, fps


# ─────────────────────────────────────────────────────────────────
# 2. MOTORE AUDIO – SINCRO TOTALE
# ─────────────────────────────────────────────────────────────────

def generate_glitch_audio(video_path, energy, scene_cuts, fps, duration, p, sr=44100):
    np.random.seed(int(p["seed"]))
    N = int(duration * sr)

    # ── Sorgente audio ──────────────────────────────────────────
    try:
        src, _ = librosa.load(video_path, sr=sr, mono=False)
        if src.ndim == 1:
            src = np.tile(src, (2, 1))
        if src.shape[1] < N:
            src = np.pad(src, ((0,0),(0, N - src.shape[1])))
        else:
            src = src[:, :N]
    except Exception:
        src = np.zeros((2, N), dtype=np.float32)

    # ── Mappa energia frame → campioni ─────────────────────────
    e_map = np.interp(
        np.linspace(0, 1, N),
        np.linspace(0, 1, len(energy)),
        energy
    ).astype(np.float32)

    out = np.zeros((2, N), dtype=np.float32)

    # ───────────────────────────────────────────────────────────
    # MECCANISMO 1 – MICRO-REAZIONE GRANULARE
    # Per ogni frame con energia > soglia bassa, emettiamo un burst
    # di grani brevi (2–20 ms) estratti ESATTAMENTE dal quel punto
    # temporale → l'audio "parla" di ciò che si vede in quel frame.
    # ───────────────────────────────────────────────────────────
    g_max = int(sr * p["g_size"])
    g_min = int(sr * 0.002)           # grano minimo: 2 ms
    intensity = p["intensity"]
    v_mix     = p["v_mix"]
    grit      = p["grit"]

    frame_times = np.linspace(0, duration, len(energy))
    active_frames = np.where(energy > 0.05)[0]   # soglia bassa: quasi tutti i frame

    for fi in active_frames:
        t    = frame_times[fi]
        i    = int(t * sr)
        pwr  = energy[fi] * intensity
        # numero di grani proporzionale all'energia (1–8)
        n_g  = max(1, int(pwr * 8))

        for _ in range(n_g):
            g_len = np.random.randint(g_min, max(g_min+1, int(g_max * pwr)))
            # piccolo jitter ±10ms attorno al frame corrente
            jitter = np.random.randint(-int(sr*0.01), int(sr*0.01)+1)
            start  = int(np.clip(i + jitter, 0, N - g_len - 1))

            grain = src[:, start:start+g_len].copy()

            # Bitcrush proporzionale al grit
            if grit > 0:
                steps = max(2, int(2 + (1.0 - grit) * 16))
                grain = np.round(grain * steps) / steps

            env = np.hanning(g_len).astype(np.float32)
            out[:, start:start+g_len] += grain * env * v_mix * (pwr + 0.05)

    # ───────────────────────────────────────────────────────────
    # MECCANISMO 2 – STUTTER
    # Sui picchi forti (energia > soglia alta) e sui scene cut,
    # prendiamo una finestra breve (stutter_len) e la ripetiamo
    # p["stutter_reps"] volte consecutive → il "freeze digitale".
    # ───────────────────────────────────────────────────────────
    stutter_len  = int(sr * p["stutter_ms"] / 1000.0)
    stutter_reps = int(p["stutter_reps"])
    peak_thresh  = float(np.percentile(energy, 85))

    # Unione picchi forti + scene cut
    peak_frames = set(np.where(energy > peak_thresh)[0].tolist())
    peak_frames.update(scene_cuts.tolist())

    written_stutter = set()
    for fi in sorted(peak_frames):
        t     = frame_times[fi]
        i     = int(t * sr)
        block = i + stutter_reps * stutter_len
        if block >= N:
            continue
        # evita sovrapposizioni ravvicinate
        if any(abs(fi - w) < 3 for w in written_stutter):
            continue
        written_stutter.add(fi)

        pwr   = energy[fi] * intensity
        chunk = src[:, i:i+stutter_len].copy()

        if grit > 0:
            steps = max(2, int(2 + (1.0 - grit) * 16))
            chunk = np.round(chunk * steps) / steps

        for rep in range(stutter_reps):
            s = i + rep * stutter_len
            e_ = min(s + stutter_len, N)
            l  = e_ - s
            env = np.hanning(l).astype(np.float32)
            # fade progressivo: ogni ripetizione è un po' più silenziosa
            decay = (1.0 - rep / stutter_reps) ** 1.5
            out[:, s:e_] += chunk[:, :l] * env * v_mix * pwr * decay

    # ───────────────────────────────────────────────────────────
    # MECCANISMO 3 – DRONE MODULATO (texture di fondo)
    # Un seno a 55 Hz la cui ampiezza segue esattamente e_map →
    # nelle zone quiete quasi scompare, nei picchi emerge.
    # ───────────────────────────────────────────────────────────
    t_ax   = np.linspace(0, duration, N, dtype=np.float32)
    drone  = np.sin(2 * np.pi * 55 * t_ax) * p["drone_vol"] * (0.1 + e_map * 0.9)
    out   += np.tile(drone, (2, 1))

    # ── Mix finale ──────────────────────────────────────────────
    mix = (src * p["v_orig_vol"]) + (out * 0.7)
    return np.clip(mix, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────
# 3. INTERFACCIA STREAMLIT
# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BeatGlitch – Sincro Totale", layout="wide")
st.title("🎬 BeatGlitch – Sincro Totale")
st.caption("Audio-visual glitch art: l'audio è la voce dell'interferenza digitale.")

with st.sidebar:
    st.header("📂 Risorse")
    v_file  = st.file_uploader("Carica Video", type=["mp4","mov"])
    st.markdown("---")
    st.subheader("💾 Preset JSON")
    up_json = st.file_uploader("Carica Preset", type=["json"])
    if st.button("📥 Prepara Download Preset"):
        defaults = {"v_orig_vol":0.25,"v_mix":2.0,"grit":0.8,"g_size":0.04,
                    "intensity":1.3,"drone_vol":0.12,
                    "stutter_ms":80,"stutter_reps":6,"seed":42}
        curr = {k: st.session_state.get(k, v) for k,v in defaults.items()}
        st.download_button("Scarica Ora", json.dumps(curr,indent=2), "preset.json")

ld = {}
if up_json:
    try: ld = json.load(up_json)
    except: st.sidebar.warning("Preset non valido.")

st.subheader("🎛️ Controlli")
c1, c2, c3 = st.columns(3)
with c1:
    v_orig_vol = st.slider("Volume Originale",      0.0, 1.0,  float(ld.get("v_orig_vol", 0.25)), key="v_orig_vol")
    v_mix      = st.slider("Potenza Glitch",        0.0, 5.0,  float(ld.get("v_mix",      2.0)),  key="v_mix")
with c2:
    grit       = st.slider("Grit (Bitcrush)",       0.0, 1.0,  float(ld.get("grit",       0.8)),  key="grit")
    g_size     = st.slider("Durata Grano (s)",      0.002,0.1, float(ld.get("g_size",     0.04)), key="g_size")
with c3:
    intensity  = st.slider("Intensità Reazione",    0.0, 2.0,  float(ld.get("intensity",  1.3)),  key="intensity")
    drone_vol  = st.slider("Drone 55 Hz",           0.0, 1.0,  float(ld.get("drone_vol",  0.12)), key="drone_vol")

st.subheader("🔁 Stutter")
cs1, cs2, cs3 = st.columns(3)
with cs1:
    stutter_ms   = st.slider("Durata Loop (ms)",   10, 300, int(ld.get("stutter_ms",   80)), key="stutter_ms")
with cs2:
    stutter_reps = st.slider("Ripetizioni Stutter", 1,  16, int(ld.get("stutter_reps",  6)), key="stutter_reps")
with cs3:
    seed = st.number_input("Seed", value=int(ld.get("seed", 42)), key="seed")

if v_file:
    if "tmp_video_path" not in st.session_state:
        t_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_v.write(v_file.read()); t_v.flush()
        st.session_state["tmp_video_path"] = t_v.name
    vpath = st.session_state["tmp_video_path"]

    if st.button("🚀 GENERA GLITCH ART", use_container_width=True):
        params = {
            "v_orig_vol": v_orig_vol, "v_mix": v_mix, "grit": grit,
            "g_size": g_size, "intensity": intensity, "drone_vol": drone_vol,
            "stutter_ms": stutter_ms, "stutter_reps": stutter_reps, "seed": seed,
        }
        try:
            prog = st.progress(0, text="[1/4] Analisi video...")
            t0 = time.time()
            energy, scene_cuts, fps = analyze_video(vpath)
            st.caption(f"✓ Frames: {len(energy)}, Scene cuts: {len(scene_cuts)}  ({time.time()-t0:.1f}s)")

            prog.progress(25, text="[2/4] Apertura clip...")
            clip     = VideoFileClip(vpath)
            duration = clip.duration
            st.caption(f"✓ Durata: {duration:.2f}s")

            prog.progress(40, text="[3/4] Sintesi glitch (micro-reazione + stutter)...")
            t0    = time.time()
            audio = generate_glitch_audio(vpath, energy, scene_cuts, fps, duration, params)
            st.caption(f"✓ Audio sintetizzato  ({time.time()-t0:.1f}s)")

            prog.progress(75, text="[4/4] Export video...")
            wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(wav_tmp.name, audio.T, 44100)
            out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            t0 = time.time()
            clip.set_audio(AudioFileClip(wav_tmp.name)).write_videofile(
                out_path, codec="libx264", audio_codec="aac", logger=None)
            st.caption(f"✓ Video scritto  ({time.time()-t0:.1f}s)")

            clip.close(); os.unlink(wav_tmp.name)
            prog.progress(100, text="✅ Completato!")
            st.success("Sincronizzazione completata!")
            st.video(out_path)
            with open(out_path,"rb") as f:
                st.download_button("💾 Scarica Video Finale", f, "glitch_video.mp4", mime="video/mp4")

        except Exception as e:
            st.error(f"Errore: {e}")
            st.exception(e)
else:
    st.info("⬆️ Carica un video nella sidebar per iniziare.")
