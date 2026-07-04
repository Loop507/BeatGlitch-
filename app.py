"""
IKEDA ENGINE — motore generativo audio/video "alla Ikeda"
Genera una PARTITURA astratta (procedurale e/o da MIDI/audio/video esterni)
e la trasforma sia in geometria visiva sia in sintesi sonora, dallo stesso dato.
"""
import numpy as np
import cv2

SR = 44100
FPS = 30

SOURCE_COLOR = {
    "ca":    (235, 235, 235),   # procedurale — bianco/grigio
    "midi":  (255, 180, 0),     # ambra/oro
    "audio": (255, 40, 140),    # magenta
    "video": (0, 210, 255),     # ciano
}

# ============================================================
# 1. GENERATORI PROCEDURALI (usati come default / fallback)
# ============================================================

def value_noise_1d(n_points, seed, octaves=5, persistence=0.55, base_freq=3):
    """Rumore smussato multi-ottava, senza dipendenze esterne (sostituisce Perlin)."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, n_points)
    signal = np.zeros(n_points)
    amp, freq, max_amp = 1.0, base_freq, 0.0
    for _ in range(octaves):
        n_anchors = max(2, int(freq))
        anchors = rng.uniform(0, 1, n_anchors)
        anchor_t = np.linspace(0, 1, n_anchors)
        signal += amp * np.interp(t, anchor_t, anchors)
        max_amp += amp
        amp *= persistence
        freq *= 2
    return signal / max_amp


def generate_macro_envelope_procedural(duration, seed, resolution=200):
    env = value_noise_1d(resolution, seed)
    return (env - env.min()) / (env.max() - env.min() + 1e-9)


def cellular_automaton(rule, width, steps, seed):
    rng = np.random.RandomState(seed)
    row = rng.randint(0, 2, width)
    row[width // 2] = 1
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)])
    grid = np.zeros((steps, width), dtype=np.uint8)
    grid[0] = row
    for i in range(1, steps):
        left, right = np.roll(row, 1), np.roll(row, -1)
        idx = (left * 4 + row * 2 + right).astype(int)
        row = rule_bits[idx]
        grid[i] = row
    return grid


def generate_events_procedural(duration, seed, rule=30, width=81, max_per_row=5):
    steps = max(20, int(duration * 8))
    grid = cellular_automaton(rule, width, steps, seed)
    events = []
    for i in range(steps):
        active = np.where(grid[i] == 1)[0]
        if len(active) == 0:
            continue
        t = (i / steps) * duration
        for a in active[:max_per_row]:
            pitch = 36 + int((a / width) * 48)
            events.append({
                "t": float(t),
                "dur": float(duration / steps) * 1.5,
                "pitch": pitch,
                "vel": 0.55,
                "source": "ca",
            })
    return events


def prime_sequence(n):
    primes, candidate = [], 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return np.array(primes)


def generate_micro_texture_procedural(n_points, seed):
    primes = prime_sequence(n_points)
    tex = (primes % 97) / 97.0
    rng = np.random.RandomState(seed)
    return tex * 0.7 + rng.uniform(0, 1, n_points) * 0.3


# ============================================================
# 2. ESTRATTORI DA INPUT ESTERNI (opzionali)
# ============================================================

def extract_from_midi(midi_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    duration = pm.get_end_time()
    events = []
    for inst_i, inst in enumerate(pm.instruments):
        for note in inst.notes:
            events.append({
                "t": float(note.start),
                "dur": float(max(0.05, note.end - note.start)),
                "pitch": int(note.pitch),
                "vel": float(note.velocity) / 127.0,
                "source": "midi",
            })
    return events, duration


def extract_from_audio(audio_path):
    import librosa
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / sr
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    events = []
    for ot in onset_times:
        idx = int(np.argmin(np.abs(rms_times - ot)))
        vel = float(min(1.0, rms[idx] / (rms.max() + 1e-9)))
        events.append({"t": float(ot), "dur": 0.15, "pitch": 60, "vel": vel, "source": "audio"})

    env = rms / (rms.max() + 1e-9)
    return events, env, duration


def extract_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    prev_gray, motion, cuts_t, frame_i = None, [], [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (80, 60))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
            motion.append(diff)
            if diff > 0.35:
                cuts_t.append(frame_i / fps)
        else:
            motion.append(0.0)
        prev_gray = gray
        frame_i += 1
    cap.release()
    duration = frame_i / fps if fps else 0
    motion = np.array(motion) if motion else np.zeros(1)
    motion = motion / (motion.max() + 1e-9)
    events = [{"t": float(t), "dur": 0.3, "pitch": 48, "vel": 0.7, "source": "video"} for t in cuts_t]
    return events, motion, duration


# ============================================================
# 3. COMBINAZIONE — costruzione della partitura condivisa
# ============================================================

def build_score(duration, seed, rule=30,
                 midi_events=None,
                 audio_events=None, audio_env=None,
                 video_events=None, video_env=None,
                 resolution=200):
    events = list(generate_events_procedural(duration, seed, rule=rule))
    for extra in (midi_events, audio_events, video_events):
        if extra:
            events += extra
    events.sort(key=lambda e: e["t"])

    macro = generate_macro_envelope_procedural(duration, seed, resolution)
    external_envs = [e for e in (audio_env, video_env) if e is not None and len(e) > 1]
    if external_envs:
        resampled = [
            np.interp(np.linspace(0, 1, resolution), np.linspace(0, 1, len(e)), e)
            for e in external_envs
        ]
        combined_ext = np.mean(resampled, axis=0)
        macro = 0.4 * macro + 0.6 * combined_ext  # gli esterni pesano di più se presenti

    texture = generate_micro_texture_procedural(resolution * 4, seed)

    return {
        "duration": duration,
        "seed": seed,
        "events": events,
        "macro_envelope": macro,
        "micro_texture": texture,
    }


# ============================================================
# 4. GENERATORE VISIVO
# ============================================================

def render_frame(t, score, width=960, height=540):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    res = len(score["macro_envelope"])
    env_idx = min(res - 1, int((t / max(score["duration"], 1e-6)) * res))
    macro_v = float(score["macro_envelope"][env_idx])

    # griglia sottile di fondo — rigore/struttura sempre visibile
    grid_alpha = int(12 + macro_v * 18)
    for gx in range(0, width, 40):
        frame[:, gx] = (grid_alpha, grid_alpha, grid_alpha)

    active = [e for e in score["events"] if e["t"] <= t <= e["t"] + max(e["dur"], 0.05)]
    for e in active:
        x = int((e["pitch"] % 96) / 96 * (width - 10))
        prog = (t - e["t"]) / max(e["dur"], 0.05)
        h = int(height * (0.15 + 0.8 * e["vel"]) * (1 - prog * 0.3))
        color = SOURCE_COLOR.get(e["source"], (255, 255, 255))
        bar_w = max(2, int(6 * (0.5 + macro_v)))
        y0 = height // 2 - h // 2
        cv2.rectangle(frame, (x, y0), (x + bar_w, y0 + max(h, 2)), color, -1)

    return frame


# ============================================================
# 5. GENERATORE AUDIO
# ============================================================

def synthesize_audio(score, sr=SR):
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    out = np.zeros(N)
    t_ax = np.linspace(0, duration, N)

    env_full = np.interp(t_ax, np.linspace(0, duration, len(score["macro_envelope"])), score["macro_envelope"])
    tex_full = np.interp(t_ax, np.linspace(0, duration, len(score["micro_texture"])), score["micro_texture"])

    # drone macro — frequenza istantanea integrata correttamente (fase continua)
    base_freq = 55.0
    drone_freq = base_freq * (1 + tex_full * 0.5)
    phase = 2 * np.pi * np.cumsum(drone_freq) / sr
    drone = np.sin(phase) * (0.06 + 0.10 * env_full)
    out += drone

    # eventi discreti — ogni fonte ha lo stesso motore di sintesi (stessa "grammatica")
    for e in score["events"]:
        start = int(e["t"] * sr)
        dur_n = max(int(0.03 * sr), int(e["dur"] * sr))
        end = min(N, start + dur_n)
        if start >= N or end <= start:
            continue
        seg_len = end - start
        freq = 440.0 * (2 ** ((e["pitch"] - 69) / 12))
        seg_t = np.arange(seg_len) / sr
        wave = np.sin(2 * np.pi * freq * seg_t) * e["vel"]
        env_local = np.hanning(seg_len) if seg_len > 1 else np.ones(seg_len)
        out[start:end] += wave * env_local * 0.45

    out = np.clip(out, -1.0, 1.0)
    return np.tile(out, (2, 1))


import os, json, tempfile
import streamlit as st
try:
    from moviepy.editor import VideoClip, AudioFileClip  # moviepy 1.x
except ModuleNotFoundError:
    from moviepy import VideoClip, AudioFileClip  # moviepy 2.x
import soundfile as sf

st.set_page_config(page_title="Ikeda Engine — Loop507", layout="wide")
st.title("◧ IKEDA ENGINE")
st.caption("Generazione audio-video procedurale da una partitura condivisa. "
           "MIDI, audio e video sono opzionali: senza input, il sistema genera da sé.")

# ------------------------------------------------------------
# SIDEBAR — sorgenti e parametri
# ------------------------------------------------------------
with st.sidebar:
    st.header("📥 Input esterni (opzionali)")
    midi_file = st.file_uploader("MIDI", type=["mid", "midi"])
    audio_file = st.file_uploader("Audio", type=["mp3", "wav"])
    video_file = st.file_uploader("Video", type=["mp4", "mov"])

    st.markdown("---")
    st.header("🎛️ Parametri generativi")
    seed = st.number_input("🎲 Seed", value=42, step=1)
    rule = st.selectbox("Regola automa cellulare", [30, 90, 110, 54, 60, 150],
                         help="Regola di Wolfram: 30=caotica, 90=frattale, 110=complessa")
    manual_duration = st.slider("Durata (se nessun input caricato)", 5, 60, 15)

    st.markdown("---")
    show_source_legend = st.checkbox("Mostra legenda colori fonte", value=True)

# ------------------------------------------------------------
# ESTRAZIONE — ogni input presente alimenta un ruolo diverso
# ------------------------------------------------------------
midi_events, audio_events, audio_env, video_events, video_env = None, None, None, None, None
durations = []

def save_upload(f, suffix):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    t.write(f.read())
    t.close()
    return t.name

if midi_file:
    with st.spinner("Estrazione eventi da MIDI..."):
        midi_path = save_upload(midi_file, ".mid")
        midi_events, midi_dur = extract_from_midi(midi_path)
        durations.append(midi_dur)
        st.sidebar.success(f"MIDI: {len(midi_events)} note estratte")

if audio_file:
    with st.spinner("Estrazione onset/energia da audio..."):
        audio_path = save_upload(audio_file, os.path.splitext(audio_file.name)[1])
        audio_events, audio_env, audio_dur = extract_from_audio(audio_path)
        durations.append(audio_dur)
        st.sidebar.success(f"Audio: {len(audio_events)} onset rilevati")

if video_file:
    with st.spinner("Analisi tagli di scena/motion da video..."):
        video_path = save_upload(video_file, ".mp4")
        video_events, video_env, video_dur = extract_from_video(video_path)
        durations.append(video_dur)
        st.sidebar.success(f"Video: {len(video_events)} tagli di scena rilevati")

duration = max(durations) if durations else float(manual_duration)

st.write(f"**Durata risultante:** {duration:.1f}s "
         f"({'da input caricati' if durations else 'da slider, nessun input caricato'})")

if show_source_legend:
    legend_cols = st.columns(4)
    labels = {"ca": "Procedurale (automa cellulare)", "midi": "MIDI",
              "audio": "Audio", "video": "Video"}
    for col, (key, label) in zip(legend_cols, labels.items()):
        r, g, b = SOURCE_COLOR[key]
        col.markdown(
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<div style='width:14px;height:14px;background:rgb({r},{g},{b});border-radius:2px'></div>"
            f"<span style='font-size:0.85em'>{label}</span></div>",
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------
# GENERAZIONE
# ------------------------------------------------------------
if st.button("🚀 GENERA", use_container_width=True):
    with st.status("Costruzione partitura e rendering...", expanded=True) as status:
        st.write("Combinazione delle fonti in un'unica partitura...")
        score = build_score(
            duration=duration, seed=int(seed), rule=rule,
            midi_events=midi_events,
            audio_events=audio_events, audio_env=audio_env,
            video_events=video_events, video_env=video_env,
        )
        st.write(f"Partitura pronta — {len(score['events'])} eventi totali.")

        st.write("Sintesi audio...")
        audio_data = synthesize_audio(score, sr=SR)
        t_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(t_wav.name, audio_data.T, SR)
        t_wav.close()

        st.write("Rendering fotogrammi (può richiedere qualche minuto)...")

        def make_frame(t):
            return render_frame(t, score)

        clip = VideoClip(make_frame, duration=score["duration"])
        clip = clip.with_fps(FPS) if hasattr(clip, "with_fps") else clip.set_fps(FPS)
        audio_clip = AudioFileClip(t_wav.name)
        clip = clip.with_audio(audio_clip) if hasattr(clip, "with_audio") else clip.set_audio(audio_clip)

        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        clip.write_videofile(out_path, codec="libx264", audio_codec="aac",
                              fps=FPS, logger=None)

        status.update(label="Fatto!", state="complete")

    st.video(out_path)
    with open(out_path, "rb") as f:
        st.download_button("💾 Scarica video", f, file_name="ikeda_output.mp4")

    preset_export = {
        "seed": int(seed), "rule": rule, "duration": score["duration"],
        "n_events": len(score["events"]),
        "sources_used": sorted(set(e["source"] for e in score["events"])),
    }
    st.sidebar.download_button("💾 Esporta partitura (info)",
                                json.dumps(preset_export, indent=2),
                                "ikeda_score_info.json")
