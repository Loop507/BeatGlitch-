"""
MATRICE — motore generativo audio/video procedurale (Loop507)
Genera una struttura dati astratta (la "matrice", procedurale e/o alimentata
da MIDI/audio/video esterni) e la trasforma sia in geometria visiva sia in
sintesi sonora, dallo stesso dato condiviso.
"""
import numpy as np
import cv2

SR = 44100
FPS = 30

# Ogni modulo della serie BeatGlitch — Matrice Engine è dedicato a un riferimento
# concettuale diverso. 01=Ikeda (rigore/automa cellulare), 02=Henke (separazione
# macro/micro). 03/04 (Molnár/Jeck) arriveranno in seguito con lo stesso schema.
MODULES = {
    "01": {
        "nome": "Ikeda — Cellular Drift",
        "effetto": "Cellular Drift",
        "processo": "Matrice Condivisa",
        "motore_tag": "generativo condiviso",
        "quote": "Un unico dato ha generato insieme cio\' che si vede e cio\' che si sente.",
        "hashtag": "#cellularautomaton #computationalminimalism",
    },
    "02": {
        "nome": "Henke — Macro/Micro Split",
        "effetto": "Macro Block Drift",
        "processo": "Struttura a Blocchi + Grana Micro",
        "motore_tag": "macro/micro separati",
        "quote": "La struttura non sfuma. Cambia di scatto, mentre la grana continua a vibrare.",
        "hashtag": "#macromicron #granularsynthesis",
    },
}

SOURCE_COLOR = {
    "ca":    (235, 235, 235),   # procedurale — bianco/grigio
    "midi":  (255, 180, 0),     # ambra/oro
    "audio": (255, 40, 140),    # magenta
    "video": (0, 210, 255),     # ciano
}

# Palette selezionabili dall'utente: se diversa da "Multicolore" sostituisce
# la colorazione per fonte con variazioni di luminosità di un unico colore.
PALETTES = {
    "Multicolore (per fonte)": None,
    "Rosso":   (235, 40, 40),
    "Blu":     (60, 140, 255),
    "Bianco":  (240, 240, 240),
    "Ambra":   (255, 170, 0),
    "Verde":   (60, 220, 120),
    "Per banda (bassi/medi/alti)": "BAND",
}

DEFAULT_BAND_COLORS = {
    "bass": (235, 40, 40),     # rosso
    "mid": (255, 170, 0),      # ambra
    "treble": (0, 200, 255),   # ciano
}


def get_event_color(e, palette_name, band_colors=None):
    setting = PALETTES.get(palette_name)
    if setting == "BAND":
        bc = band_colors or DEFAULT_BAND_COLORS
        base = bc.get(e.get("band"), (235, 235, 235))  # fonti senza banda -> grigio chiaro
    elif setting is None:  # multicolore per fonte
        base = SOURCE_COLOR.get(e["source"], (255, 255, 255))
    else:
        base = setting
    factor = 0.45 + 0.55 * e.get("vel", 0.6)  # più intenso = più luminoso
    return tuple(min(255, int(c * factor)) for c in base)

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


def generate_macro_blocks(duration, seed, block_seconds=3.0):
    """Struttura macro a GRADINI discreti (usata dal modulo Henke): ogni blocco ha un
    livello fisso, il cambio tra blocchi è netto — non un'interpolazione continua."""
    n_blocks = max(1, int(np.ceil(duration / block_seconds)))
    rng = np.random.RandomState(seed + 777)
    values = rng.uniform(0.15, 1.0, n_blocks)
    return {"block_seconds": block_seconds, "values": values}


def macro_block_value(macro_blocks, t):
    idx = min(len(macro_blocks["values"]) - 1, int(t // macro_blocks["block_seconds"]))
    return float(macro_blocks["values"][idx])


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


def generate_events_procedural(duration, seed, rule=30, width=81, max_per_row=5, density_env=None):
    """density_env: array opzionale 0..1 che modula nel tempo quanti eventi procedurali
    vengono mantenuti (usato per far reagire la texture di base alla dinamica dell'audio,
    invece di essere sempre identica indipendentemente dal brano caricato)."""
    steps = max(20, int(duration * 8))
    grid = cellular_automaton(rule, width, steps, seed)
    events = []
    for i in range(steps):
        active = np.where(grid[i] == 1)[0]
        if len(active) == 0:
            continue
        t = (i / steps) * duration
        if density_env is not None:
            idx = min(len(density_env) - 1, int((i / steps) * len(density_env)))
            row_cap = max(1, int(max_per_row * density_env[idx]))
        else:
            row_cap = max_per_row
        for a in active[:row_cap]:
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


BAND_PARAMS = {
    "bass":   {"dur": 0.35, "thickness": 1.9, "pitch": 40},
    "mid":    {"dur": 0.15, "thickness": 1.0, "pitch": 62},
    "treble": {"dur": 0.05, "thickness": 0.55, "pitch": 84},
}


def _dominant_band(y_window, sr):
    """Determina quale banda (bassi/medi/alti) domina in una finestra audio."""
    if len(y_window) < 8:
        return "mid"
    spectrum = np.abs(np.fft.rfft(y_window))
    freqs = np.fft.rfftfreq(len(y_window), 1.0 / sr)
    bass = spectrum[(freqs >= 20) & (freqs < 250)].sum()
    mid = spectrum[(freqs >= 250) & (freqs < 2000)].sum()
    treble = spectrum[(freqs >= 2000) & (freqs < 8000)].sum()
    energies = {"bass": bass, "mid": mid, "treble": treble}
    return max(energies, key=energies.get)


def _pan_of_window(l_window, r_window):
    """Bilanciamento stereo -1 (tutto a sinistra) .. +1 (tutto a destra)."""
    l_energy = float(np.sum(np.abs(l_window)))
    r_energy = float(np.sum(np.abs(r_window)))
    total = l_energy + r_energy
    if total < 1e-9:
        return 0.0
    return float(np.clip((r_energy - l_energy) / total, -1.0, 1.0))


def extract_from_audio(audio_path):
    import librosa
    y_raw, sr = librosa.load(audio_path, sr=SR, mono=False)
    if y_raw.ndim == 1:
        y_stereo = np.stack([y_raw, y_raw])  # mono duplicato: pan sempre 0 (centro)
    else:
        y_stereo = y_raw[:2]
    y_mono = y_stereo.mean(axis=0)
    duration = y_mono.shape[0] / sr

    onset_frames = librosa.onset.onset_detect(y=y_mono, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    rms = librosa.feature.rms(y=y_mono)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    win = 2048
    events = []
    for ot in onset_times:
        idx = int(np.argmin(np.abs(rms_times - ot)))
        vel = float(min(1.0, rms[idx] / (rms.max() + 1e-9)))

        sample_pos = int(ot * sr)
        lo, hi = max(0, sample_pos - win // 2), sample_pos + win // 2
        window_mono = y_mono[lo:hi]
        band = _dominant_band(window_mono, sr)
        bp = BAND_PARAMS[band]
        pan = _pan_of_window(y_stereo[0, lo:hi], y_stereo[1, lo:hi])

        events.append({
            "t": float(ot), "dur": bp["dur"], "pitch": bp["pitch"] + int(vel * 8),
            "vel": vel, "source": "audio", "band": band, "pan": pan,
        })

    env = rms / (rms.max() + 1e-9)
    return events, env, duration, y_stereo


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
# 3. COMBINAZIONE — costruzione della matrice condivisa
# ============================================================

def _content_seed_offset(audio_env, video_env, midi_events):
    """Deriva un offset di seed dal contenuto reale delle fonti esterne, così brani/video
    diversi con lo stesso seed utente producono comunque un automa cellulare diverso —
    invece che sempre lo stesso, con solo pochi eventi colorati sparsi sopra."""
    acc = 0.0
    if audio_env is not None and len(audio_env) > 0:
        acc += float(np.sum(audio_env) * 997)
    if video_env is not None and len(video_env) > 0:
        acc += float(np.sum(video_env) * 613)
    if midi_events:
        acc += sum(e["pitch"] for e in midi_events) * 31
    return int(acc) % 100000


def build_score(duration, seed, rule=30,
                 midi_events=None,
                 audio_events=None, audio_env=None,
                 video_events=None, video_env=None,
                 resolution=200, macro_block_seconds=3.0):
    has_external = bool(midi_events or audio_events or video_events)

    # il seed "di base" pilotato dall'utente resta riproducibile, ma viene perturbato
    # dal contenuto reale delle fonti esterne, in modo che due brani diversi diano
    # davvero pattern procedurali diversi, non solo pochi eventi colorati in più
    effective_seed = seed + _content_seed_offset(audio_env, video_env, midi_events)

    macro = generate_macro_envelope_procedural(duration, effective_seed, resolution)
    macro_blocks = generate_macro_blocks(duration, effective_seed, block_seconds=macro_block_seconds)
    external_envs = [e for e in (audio_env, video_env) if e is not None and len(e) > 1]
    silence_envelope = None  # se presente, pilota il "gate silenzio" nel rendering
    if external_envs:
        resampled = [
            np.interp(np.linspace(0, 1, resolution), np.linspace(0, 1, len(e)), e)
            for e in external_envs
        ]
        combined_ext = np.mean(resampled, axis=0)
        silence_envelope = combined_ext  # inviluppo reale, non mescolato col rumore procedurale
        macro = 0.4 * macro + 0.6 * combined_ext  # gli esterni pesano di più se presenti

    # quando ci sono fonti esterne, la texture procedurale fa da "base" più leggera,
    # modulata nel tempo dalla stessa dinamica del brano/video (non più costante fissa)
    max_per_row = 2 if has_external else 5
    density_env = macro if has_external else None
    events = list(generate_events_procedural(
        duration, effective_seed, rule=rule, max_per_row=max_per_row, density_env=density_env
    ))
    for extra in (midi_events, audio_events, video_events):
        if extra:
            events += extra
    events.sort(key=lambda e: e["t"])

    texture = generate_micro_texture_procedural(resolution * 4, effective_seed)

    return {
        "duration": duration,
        "seed": effective_seed,
        "macro_blocks": macro_blocks,
        "events": events,
        "macro_envelope": macro,
        "silence_envelope": silence_envelope,
        "micro_texture": texture,
    }


# ============================================================
# 4. GENERATORE VISIVO
# ============================================================

def _lane_fraction(e):
    """Posizione 0..1 lungo l'asse delle corsie: usa il pan stereo reale se disponibile
    (eventi audio), altrimenti il pitch come proxy deterministico per le altre fonti."""
    if "pan" in e:
        return (e["pan"] + 1.0) / 2.0  # da -1..1 a 0..1
    return (e["pitch"] % 96) / 96.0


def _event_orientation(e, mode):
    """In modalità 'misto' (V+H) ogni evento sceglie il proprio asse: i bassi restano
    verticali (sostenuti), gli alti orizzontali (rapidi), i medi si alternano nel tempo."""
    if mode != "misto":
        return mode
    band = e.get("band")
    if band == "bass":
        return "verticale"
    if band == "treble":
        return "orizzontale"
    return "verticale" if int(e["t"] * 10) % 2 == 0 else "orizzontale"


def render_frame_ikeda(t, score, width=960, height=540, orientation="verticale",
                        num_lanes=10, palette="Multicolore (per fonte)", band_colors=None):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    res = len(score["macro_envelope"])
    env_idx = min(res - 1, int((t / max(score["duration"], 1e-6)) * res))
    macro_v = float(score["macro_envelope"][env_idx])

    # griglia sottile di fondo — in modalità mista mostra entrambi gli assi
    grid_alpha = int(12 + macro_v * 18)
    if orientation in ("verticale", "misto"):
        grid_step = max(8, width // 24)
        for gx in range(0, width, grid_step):
            frame[:, gx] = (grid_alpha, grid_alpha, grid_alpha)
    if orientation in ("orizzontale", "misto"):
        grid_step = max(8, height // 24)
        for gy in range(0, height, grid_step):
            frame[gy, :] = (grid_alpha, grid_alpha, grid_alpha)

    # gate silenzio: durante il silenzio reale (inviluppo audio/video esterno vicino
    # a zero) non deve comparire nessuna striscia — uso l'inviluppo "puro", non la
    # matrice mescolata col rumore procedurale, altrimenti il silenzio non è mai vero zero
    SILENCE_THRESHOLD = 0.06
    gate_env = score.get("silence_envelope")
    if gate_env is not None:
        gate_idx = min(len(gate_env) - 1, int((t / max(score["duration"], 1e-6)) * len(gate_env)))
        if float(gate_env[gate_idx]) < SILENCE_THRESHOLD:
            return frame  # silenzio: solo la griglia di fondo, nessuna striscia

    active = [e for e in score["events"] if e["t"] <= t <= e["t"] + max(e["dur"], 0.05)]
    num_lanes = max(1, int(num_lanes))

    for e in active:
        prog = (t - e["t"]) / max(e["dur"], 0.05)
        color = get_event_color(e, palette, band_colors)

        band_thickness = BAND_PARAMS[e["band"]]["thickness"] if "band" in e else \
            (1.0 if e["source"] == "ca" else 1.6)
        # lo spessore ora riflette SIA la banda SIA l'intensità del colpo (il "beat")
        intensity_factor = 0.5 + 0.9 * e["vel"]
        thickness_factor = band_thickness * intensity_factor

        lane_frac = min(0.999, max(0.0, _lane_fraction(e)))
        lane_idx = int(lane_frac * num_lanes)

        # barre più grandi di default: riempiono meglio la scena e calano più lentamente
        extent_frac = (0.45 + 0.55 * e["vel"]) * (1 - prog * 0.15)
        event_orientation = _event_orientation(e, orientation)

        if event_orientation == "verticale":
            lane_w = width / num_lanes
            x_center = lane_idx * lane_w + lane_w / 2
            bar_w = int(np.clip(lane_w * 0.75 * thickness_factor, 4, lane_w * 0.98))
            bar_len = int(height * extent_frac)
            y0 = height // 2 - bar_len // 2
            x0 = int(x_center - bar_w / 2)
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + max(bar_len, 2)), color, -1)
        else:  # orizzontale
            lane_h = height / num_lanes
            y_center = lane_idx * lane_h + lane_h / 2
            bar_h = int(np.clip(lane_h * 0.75 * thickness_factor, 4, lane_h * 0.98))
            bar_len = int(width * extent_frac)
            x0 = width // 2 - bar_len // 2
            y0 = int(y_center - bar_h / 2)
            cv2.rectangle(frame, (x0, y0), (x0 + max(bar_len, 2), y0 + bar_h), color, -1)

    return frame


def render_frame_henke(t, score, width=960, height=540, orientation="verticale",
                        num_lanes=10, palette="Multicolore (per fonte)", band_colors=None):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    res = len(score["macro_envelope"])
    env_idx = min(res - 1, int((t / max(score["duration"], 1e-6)) * res))
    macro_v = float(score["macro_envelope"][env_idx])

    # griglia sottile di fondo — in modalità mista mostra entrambi gli assi
    grid_alpha = int(12 + macro_v * 18)
    if orientation in ("verticale", "misto"):
        grid_step = max(8, width // 24)
        for gx in range(0, width, grid_step):
            frame[:, gx] = (grid_alpha, grid_alpha, grid_alpha)
    if orientation in ("orizzontale", "misto"):
        grid_step = max(8, height // 24)
        for gy in range(0, height, grid_step):
            frame[gy, :] = (grid_alpha, grid_alpha, grid_alpha)

    # STRATO MACRO (Henke): una banda strutturale a gradini, che cambia di scatto
    # ai confini di blocco — mai fusa/interpolata col micro. Resta visibile ANCHE
    # durante il silenzio: è la struttura sostenuta, non la reazione istantanea.
    block_val = macro_block_value(score["macro_blocks"], t)
    macro_gray = int(35 + block_val * 55)
    if orientation == "orizzontale":
        band_w = int(width * (0.10 + 0.20 * block_val))
        x0 = width // 2 - band_w // 2
        cv2.rectangle(frame, (x0, 0), (x0 + band_w, height), (macro_gray, macro_gray, macro_gray), -1)
    else:
        band_h = int(height * (0.10 + 0.20 * block_val))
        y0 = height // 2 - band_h // 2
        cv2.rectangle(frame, (0, y0), (width, y0 + band_h), (macro_gray, macro_gray, macro_gray), -1)

    # gate silenzio: durante il silenzio reale lo strato MICRO non deve comparire —
    # la macro resta comunque visibile (è la separazione netta macro/micro di Henke)
    SILENCE_THRESHOLD = 0.06
    gate_env = score.get("silence_envelope")
    if gate_env is not None:
        gate_idx = min(len(gate_env) - 1, int((t / max(score["duration"], 1e-6)) * len(gate_env)))
        if float(gate_env[gate_idx]) < SILENCE_THRESHOLD:
            return frame  # silenzio: griglia + banda macro, ma nessun grano micro

    active = [e for e in score["events"] if e["t"] <= t <= e["t"] + max(e["dur"], 0.05)]
    num_lanes = max(1, int(num_lanes))

    for e in active:
        prog = (t - e["t"]) / max(e["dur"], 0.05)
        color = get_event_color(e, palette, band_colors)

        band_thickness = BAND_PARAMS[e["band"]]["thickness"] if "band" in e else \
            (1.0 if e["source"] == "ca" else 1.6)
        # lo spessore riflette SOLO banda+intensità del colpo — mai il valore macro,
        # per rispettare la separazione netta macro/micro
        intensity_factor = 0.5 + 0.9 * e["vel"]
        thickness_factor = band_thickness * intensity_factor

        lane_frac = min(0.999, max(0.0, _lane_fraction(e)))
        lane_idx = int(lane_frac * num_lanes)

        extent_frac = (0.45 + 0.55 * e["vel"]) * (1 - prog * 0.15)
        event_orientation = _event_orientation(e, orientation)

        if event_orientation == "verticale":
            lane_w = width / num_lanes
            x_center = lane_idx * lane_w + lane_w / 2
            bar_w = int(np.clip(lane_w * 0.75 * thickness_factor, 4, lane_w * 0.98))
            bar_len = int(height * extent_frac)
            y0 = height // 2 - bar_len // 2
            x0 = int(x_center - bar_w / 2)
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + max(bar_len, 2)), color, -1)
        else:  # orizzontale
            lane_h = height / num_lanes
            y_center = lane_idx * lane_h + lane_h / 2
            bar_h = int(np.clip(lane_h * 0.75 * thickness_factor, 4, lane_h * 0.98))
            bar_len = int(width * extent_frac)
            x0 = width // 2 - bar_len // 2
            y0 = int(y_center - bar_h / 2)
            cv2.rectangle(frame, (x0, y0), (x0 + max(bar_len, 2), y0 + bar_h), color, -1)

    return frame


# ============================================================
# 5. GENERATORE AUDIO
# ============================================================

def synthesize_audio_ikeda(score, sr=SR):
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    out_l = np.zeros(N)
    out_r = np.zeros(N)
    t_ax = np.linspace(0, duration, N)

    env_full = np.interp(t_ax, np.linspace(0, duration, len(score["macro_envelope"])), score["macro_envelope"])
    tex_full = np.interp(t_ax, np.linspace(0, duration, len(score["micro_texture"])), score["micro_texture"])

    # drone macro — frequenza istantanea integrata correttamente (fase continua), centrato
    base_freq = 55.0
    drone_freq = base_freq * (1 + tex_full * 0.5)
    phase = 2 * np.pi * np.cumsum(drone_freq) / sr
    drone = np.sin(phase) * (0.06 + 0.10 * env_full)
    out_l += drone
    out_r += drone

    # eventi discreti — ogni fonte ha lo stesso motore di sintesi (stessa "grammatica"),
    # posizionati in stereo secondo il pan reale (0.0 = centro, per fonti senza pan noto)
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
        signal = wave * env_local * 0.45

        pan = e.get("pan", 0.0)
        gain_l = float(np.sqrt((1.0 - pan) / 2.0))
        gain_r = float(np.sqrt((1.0 + pan) / 2.0))
        out_l[start:end] += signal * gain_l
        out_r[start:end] += signal * gain_r

    stereo = np.stack([np.clip(out_l, -1.0, 1.0), np.clip(out_r, -1.0, 1.0)])
    return stereo


def synthesize_audio_henke(score, sr=SR):
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    out_l = np.zeros(N)
    out_r = np.zeros(N)
    t_ax = np.linspace(0, duration, N)

    tex_full = np.interp(t_ax, np.linspace(0, duration, len(score["micro_texture"])), score["micro_texture"])

    # drone MACRO (Henke): livello a gradini discreti, uno per blocco — cambia di scatto
    # ai confini (con una brevissima rampa di pochi ms solo per evitare click, non per
    # sfumare gradualmente: la transizione resta percepibile come taglio netto)
    mb = score["macro_blocks"]
    block_idx = np.minimum(len(mb["values"]) - 1, (t_ax // mb["block_seconds"]).astype(int))
    block_level = mb["values"][block_idx]
    ramp_samples = max(1, int(0.015 * sr))  # ~15ms, solo anti-click
    if ramp_samples > 1:
        kernel = np.ones(ramp_samples) / ramp_samples
        block_level = np.convolve(block_level, kernel, mode="same")

    base_freq = 55.0
    drone_freq = base_freq * (1 + block_level * 1.2)  # la frequenza segue il gradino, non il rumore
    phase = 2 * np.pi * np.cumsum(drone_freq) / sr
    drone = np.sin(phase) * (0.05 + 0.10 * block_level)
    out_l += drone
    out_r += drone

    # eventi discreti — ogni fonte ha lo stesso motore di sintesi (stessa "grammatica"),
    # posizionati in stereo secondo il pan reale (0.0 = centro, per fonti senza pan noto)
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
        signal = wave * env_local * 0.45

        # grana granulare (Henke: la texture micro non è solo tono puro ma ha "grain")
        tex_val = float(tex_full[min(start, N - 1)])
        grain_seed = int((e["t"] * 1000) % 100000)
        noise_grain = np.random.RandomState(grain_seed).uniform(-1, 1, seg_len)
        signal = signal + noise_grain * env_local * tex_val * e["vel"] * 0.12

        pan = e.get("pan", 0.0)
        gain_l = float(np.sqrt((1.0 - pan) / 2.0))
        gain_r = float(np.sqrt((1.0 + pan) / 2.0))
        out_l[start:end] += signal * gain_l
        out_r[start:end] += signal * gain_r

    stereo = np.stack([np.clip(out_l, -1.0, 1.0), np.clip(out_r, -1.0, 1.0)])
    return stereo


def fit_audio_length(y, N):
    """Adatta un array audio (mono 1D o stereo (2,N)) alla lunghezza N (trim o loop)."""
    if y.ndim == 1:
        if len(y) == 0:
            return np.zeros(N)
        if len(y) >= N:
            return y[:N]
        return np.tile(y, int(np.ceil(N / len(y))))[:N]
    else:
        length = y.shape[1]
        if length == 0:
            return np.zeros((y.shape[0], N))
        if length >= N:
            return y[:, :N]
        reps = int(np.ceil(N / length))
        return np.tile(y, (1, reps))[:, :N]


def generate_text_report(params, score, module_id="01", brand="Loop507", vol=None):
    """Report nel formato standard Loop507 (:: MOTORE / EFFETTO / TECHNICAL LOG SHEET)."""
    meta = MODULES[module_id]
    counts = {}
    for e in score["events"]:
        counts[e["source"]] = counts.get(e["source"], 0) + 1

    band_counts = {}
    for e in score["events"]:
        if "band" in e:
            band_counts[e["band"]] = band_counts.get(e["band"], 0) + 1

    labels = {"ca": "Procedurale", "midi": "MIDI", "audio": "Audio", "video": "Video"}
    fonti_attive = [labels[k] for k in ("midi", "audio", "video") if counts.get(k, 0) > 0]

    analisi = ["Automa Cellulare", "Rumore Multi-Ottava"]
    if counts.get("audio", 0) > 0:
        analisi += ["Onset Detection", "Band Split (bassi/medi/alti)", "RMS Envelope"]
    if counts.get("video", 0) > 0:
        analisi.append("Scene Cut Detection")
    if counts.get("midi", 0) > 0:
        analisi.append("MIDI Parsing")
    if module_id == "02":
        analisi.append("Macro Block Segmentation")

    vol_num = vol if vol is not None else abs(score["seed"]) % 99
    n_frames = int(round(score["duration"] * FPS))

    righe = []
    righe.append(f"[BEATGLITCH_MATRICE_ENGINE_{module_id}] // VOL_{vol_num:02d} // H.264 // DATA_FRAGMENT")
    righe.append(f":: MOTORE: matrice_engine_{module_id} [v1.0 — {meta['motore_tag']}]")
    righe.append(f":: EFFETTO: {meta['effetto']} — Regola {params['rule']}")
    righe.append(f":: ANALISI: {' / '.join(analisi)}")
    fonti_str = " + ".join(fonti_attive) if fonti_attive else "Nessuna (generazione pura)"
    righe.append(f":: PROCESSO: {meta['processo']} — Fonti: {fonti_str}")
    righe.append("")
    righe.append(f'"{meta["quote"]}"')
    righe.append("")
    righe.append("> TECHNICAL LOG SHEET:")
    righe.append(f"* File: matrice_output_{params['timestamp'].replace('/', '').replace(':', '').replace(' ', '_')}")
    righe.append(f"* Modulo: {module_id} — {meta['nome']}")
    righe.append(f"* Seed (utente): {params['seed']}")
    righe.append(f"* Seed effettivo: {score['seed']} (perturbato dal contenuto delle fonti esterne)")
    righe.append(f"* Rendering: {n_frames} frame @ {FPS}fps")
    righe.append(f"* Risoluzione: {params['resolution']}")
    righe.append(f"* Durata: {score['duration']:.1f}s")
    if module_id == "02":
        n_blocks = len(score["macro_blocks"]["values"])
        righe.append(f"* Blocchi macro: {n_blocks} (ogni {score['macro_blocks']['block_seconds']:.1f}s)")
    righe.append(f"* Colonna sonora: {params['audio_mode']}")
    righe.append(f"* Eventi totali: {len(score['events'])}")
    righe.append(f"* Eventi Procedurali: {counts.get('ca', 0)}")
    if counts.get("midi", 0) > 0:
        righe.append(f"* Eventi MIDI: {counts.get('midi', 0)}")
    if counts.get("audio", 0) > 0:
        righe.append(f"* Eventi Audio: {counts.get('audio', 0)}")
        righe.append(f"* Bilanciamento Frequenze: bassi {band_counts.get('bass', 0)} | "
                      f"medi {band_counts.get('mid', 0)} | alti {band_counts.get('treble', 0)}")
    if counts.get("video", 0) > 0:
        righe.append(f"* Eventi Video (tagli scena): {counts.get('video', 0)}")
    righe.append("")
    righe.append(f"> Regia e Algoritmo: {brand}")
    righe.append("")
    righe.append(f"#generativeart #proceduralart #digitalminimalism {meta['hashtag']}")
    righe.append("#computationalminimalism #brutalistart #glitchart #audiovisual")
    righe.append("#experimentalvideo #beatglitch")

    return "\n".join(righe)






import os, json, tempfile, base64
from datetime import datetime
import streamlit as st
try:
    from moviepy.editor import VideoClip, AudioFileClip  # moviepy 1.x
except ModuleNotFoundError:
    from moviepy import VideoClip, AudioFileClip  # moviepy 2.x
import soundfile as sf

st.set_page_config(page_title="BeatGlitch — Matrice Engine", layout="wide")
st.title("◧ BEATGLITCH — MATRICE ENGINE")
st.caption("Generazione audio-video procedurale da una struttura dati condivisa. "
           "MIDI, audio e video sono opzionali: senza input, il sistema genera da sé.")

modulo_label = st.radio(
    "Modulo",
    [f"{mid} — {MODULES[mid]['nome']}" for mid in MODULES],
    horizontal=True,
)
module_id = modulo_label.split(" — ")[0]
MODULE_TITLE = f"BEATGLITCH — MATRICE ENGINE {module_id}"
MODULE_FILENAME_BASE = f"beatglitch_matrice_engine_{module_id}"
render_frame_fn = render_frame_ikeda if module_id == "01" else render_frame_henke
synthesize_audio_fn = synthesize_audio_ikeda if module_id == "01" else synthesize_audio_henke
if module_id == "02":
    st.caption("Modulo Henke: la banda macro (struttura a blocchi) resta visibile anche "
               "nel silenzio — solo i grani micro reagiscono all'istante presente.")

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
    st.header("📐 Formato export")
    formato_export = st.selectbox("Dimensioni video", [
        "1280x720 (16:9)", "720x1280 (9:16)", "720x720 (1:1)"
    ])
    EXPORT_SIZES = {
        "1280x720 (16:9)": (1280, 720),
        "720x1280 (9:16)": (720, 1280),
        "720x720 (1:1)": (720, 720),
    }
    export_w, export_h = EXPORT_SIZES[formato_export]

    st.markdown("---")
    st.header("🎨 Aspetto visivo")
    orientamento_label = st.radio("Orientamento linee", ["Verticali", "Orizzontali", "Verticali + Orizzontali"],
                                   horizontal=True)
    ORIENTAMENTI = {"Verticali": "verticale", "Orizzontali": "orizzontale",
                     "Verticali + Orizzontali": "misto"}
    orientamento = ORIENTAMENTI[orientamento_label]
    if orientamento == "misto":
        st.caption("In modalità mista: bassi verticali (sostenuti), alti orizzontali (rapidi), "
                   "medi alternati nel tempo.")
    num_lanes = st.slider("Numero di linee", 1, 24, 10)
    palette = st.selectbox("Palette colore", list(PALETTES.keys()))

    band_colors = None
    if palette == "Per banda (bassi/medi/alti)":
        st.caption("Un colore per ciascuna banda di frequenza (solo eventi audio; "
                   "le altre fonti restano grigio chiaro).")
        c_bassi = st.color_picker("Bassi", "#EB2828")
        c_medi = st.color_picker("Medi", "#FFAA00")
        c_alti = st.color_picker("Alti", "#00C8FF")

        def _hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

        band_colors = {"bass": _hex_to_rgb(c_bassi), "mid": _hex_to_rgb(c_medi),
                        "treble": _hex_to_rgb(c_alti)}

    st.markdown("---")
    show_source_legend = st.checkbox("Mostra legenda colori fonte", value=True,
                                      disabled=(palette != "Multicolore (per fonte)"))

# ------------------------------------------------------------
# ESTRAZIONE — ogni input presente alimenta un ruolo diverso
# ------------------------------------------------------------
midi_events, audio_events, audio_env, video_events, video_env = None, None, None, None, None
audio_raw = None
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
        audio_events, audio_env, audio_dur, audio_raw = extract_from_audio(audio_path)
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

if show_source_legend and palette == "Multicolore (per fonte)":
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
# COLONNA SONORA FINALE
# ------------------------------------------------------------
audio_mode_options = ["Generata (sintesi)"]
if audio_raw is not None:
    audio_mode_options += ["Originale (file caricato)", "Mix (generata + originale)"]
audio_mode = st.radio("🎧 Colonna sonora finale", audio_mode_options, horizontal=True)
st.caption("Nota: questo switch cambia solo l'audio che senti. Il video reagisce sempre "
           "alla stessa matrice (gli stessi eventi), indipendentemente da quale "
           "colonna sonora scegli per il render finale.")

# ------------------------------------------------------------
# GENERAZIONE
# ------------------------------------------------------------
if st.button("🚀 GENERA", use_container_width=True):
    with st.status("Costruzione matrice e rendering...", expanded=True) as status:
        st.write("Combinazione delle fonti in un'unica matrice...")
        score = build_score(
            duration=duration, seed=int(seed), rule=rule,
            midi_events=midi_events,
            audio_events=audio_events, audio_env=audio_env,
            video_events=video_events, video_env=video_env,
        )
        st.write(f"Matrice pronta — {len(score['events'])} eventi totali.")

        st.write("Costruzione colonna sonora finale...")
        N = int(score["duration"] * SR)
        generated = synthesize_audio_fn(score, sr=SR)  # shape (2, N)

        if audio_mode == "Originale (file caricato)" and audio_raw is not None:
            final_audio = fit_audio_length(audio_raw, N)
        elif audio_mode == "Mix (generata + originale)" and audio_raw is not None:
            original_stereo = fit_audio_length(audio_raw, N)
            final_audio = np.clip(generated * 0.6 + original_stereo * 0.6, -1.0, 1.0)
        else:
            final_audio = generated

        t_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(t_wav.name, final_audio.T, SR)
        t_wav.close()

        st.write(f"Rendering fotogrammi a {export_w}x{export_h} (può richiedere qualche minuto)...")

        def make_frame(t):
            return render_frame_fn(t, score, width=export_w, height=export_h,
                                    orientation=orientamento, num_lanes=num_lanes,
                                    palette=palette, band_colors=band_colors)

        clip = VideoClip(make_frame, duration=score["duration"])
        clip = clip.with_fps(FPS) if hasattr(clip, "with_fps") else clip.set_fps(FPS)
        audio_clip = AudioFileClip(t_wav.name)
        clip = clip.with_audio(audio_clip) if hasattr(clip, "with_audio") else clip.set_audio(audio_clip)

        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        clip.write_videofile(out_path, codec="libx264", audio_codec="aac",
                              fps=FPS, logger=None)

        st.write("Generazione report...")
        ts_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{MODULE_FILENAME_BASE}_{ts_compact}"
        report_params = {
            "seed": int(seed), "rule": rule, "duration": score["duration"],
            "resolution": f"{export_w}x{export_h}", "audio_mode": audio_mode,
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
        report_text = generate_text_report(report_params, score, module_id=module_id)

        status.update(label="Fatto!", state="complete")

    # salvo tutto in session_state: sopravvive ai rerun causati dai download_button,
    # così video e report non spariscono l'uno cliccando sull'altro
    st.session_state["result"] = {
        "video_path": out_path,
        "report_text": report_text,
        "base_filename": base_filename,
        "preset_export": {
            "seed": int(seed), "rule": rule, "duration": score["duration"],
            "n_events": len(score["events"]),
            "sources_used": sorted(set(e["source"] for e in score["events"])),
            "risoluzione": f"{export_w}x{export_h}",
            "colonna_sonora": audio_mode,
        },
    }

# ------------------------------------------------------------
# VISUALIZZAZIONE RISULTATI — fuori dal blocco del pulsante,
# così un download non fa sparire l'altro
# ------------------------------------------------------------
if "result" in st.session_state:
    res = st.session_state["result"]

    # embed diretto con larghezza fissa in pixel — più affidabile del CSS su st.video,
    # che su alcune versioni di Streamlit viene sovrascritto dallo stile di default
    with open(res["video_path"], "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div style="text-align:center">'
        f'<video width="360" controls src="data:video/mp4;base64,{video_b64}"></video>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_dl1, col_dl2 = st.columns(2)
    with open(res["video_path"], "rb") as f:
        col_dl1.download_button("💾 Scarica video", f, file_name=f"{res['base_filename']}.mp4",
                                 use_container_width=True, key="dl_video")
    col_dl2.download_button("📄 Scarica report (txt)", res["report_text"],
                             file_name=f"{res['base_filename']}_report.txt", mime="text/plain",
                             use_container_width=True, key="dl_report")

    st.sidebar.download_button("💾 Esporta matrice (info)",
                                json.dumps(res["preset_export"], indent=2),
                                f"{res['base_filename']}_info.json", key="dl_json")
