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
