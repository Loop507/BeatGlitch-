"""
MATRICE — motore generativo audio/video procedurale (Loop507)
Genera una struttura dati astratta (la "matrice", procedurale e/o alimentata
da MIDI/audio/video esterni) e la trasforma sia in geometria visiva sia in
sintesi sonora, dallo stesso dato condiviso.
"""
import numpy as np
import cv2
import os
import json
import tempfile
from functools import lru_cache

SR = 44100
FPS = 30

# Ogni modulo della serie BeatGlitch — Matrice Engine è dedicato a un riferimento
# concettuale diverso. 01=Ikeda (rigore/automa cellulare), 02=Henke (separazione
# macro/micro), 03=Molnár (disordine controllato), 04=Jeck (usura con memoria).
MODULES = {
    "01": {
        "nome": "Matrice 01 — Cellular Drift",
        "nome_en": "Matrix 01 — Cellular Drift",
        "effetto": "Cellular Drift",
        "processo": "Matrice Condivisa",
        "processo_en": "Shared Matrix",
        "motore_tag": "generativo condiviso",
        "motore_tag_en": "shared generative",
        "quote": "Un unico dato ha generato insieme cio\' che si vede e cio\' che si sente.",
        "quote_en": "A single piece of data generated what you see and what you hear together.",
        "hashtag": "#cellularautomaton #computationalminimalism",
    },
    "02": {
        "nome": "Matrice 02 — Macro/Micro Split",
        "nome_en": "Matrix 02 — Macro/Micro Split",
        "effetto": "Macro Block Drift",
        "processo": "Struttura a Blocchi + Grana Micro",
        "processo_en": "Block Structure + Micro Grain",
        "motore_tag": "macro/micro separati",
        "motore_tag_en": "separate macro/micro",
        "quote": "La struttura non sfuma. Cambia di scatto, mentre la grana continua a vibrare.",
        "quote_en": "The structure never fades. It cuts abruptly, while the grain keeps vibrating.",
        "hashtag": "#macromicron #granularsynthesis",
    },
    "03": {
        "nome": "Matrice 03 — Disordine Controllato",
        "nome_en": "Matrix 03 — Controlled Disorder",
        "effetto": "Grid Deviation",
        "processo": "Griglia Rigida + Deviazioni Rare",
        "processo_en": "Rigid Grid + Rare Deviations",
        "motore_tag": "griglia + eccezioni rare",
        "motore_tag_en": "grid + rare exceptions",
        "quote": "La regola resta quasi sempre uguale. Quando si rompe, lo fa apposta.",
        "quote_en": "The rule stays almost always the same. When it breaks, it does so on purpose.",
        "hashtag": "#griddeviation #controlleddisorder",
    },
    "04": {
        "nome": "Matrice 04 — Usura con Memoria",
        "nome_en": "Matrix 04 — Wear with Memory",
        "effetto": "Tape Decay",
        "processo": "Degrado Cumulativo Persistente",
        "processo_en": "Persistent Cumulative Decay",
        "motore_tag": "usura con memoria",
        "motore_tag_en": "wear with memory",
        "quote": "Non reagisce solo a ora. Ricorda ogni volta che e' stato suonato prima.",
        "quote_en": "It doesn't only react to now. It remembers every time it was played before.",
        "hashtag": "#tapedecay #memorydecay",
    },
    "05": {
        "nome": "Matrice 05 — Campi Stocastici",
        "nome_en": "Matrix 05 — Stochastic Fields",
        "effetto": "Stochastic Field Sweep",
        "processo": "Fasci di Linee Divergenti + Pulviscolo Stocastico",
        "processo_en": "Divergent Line Bundles + Stochastic Dust",
        "motore_tag": "campo continuo stocastico",
        "motore_tag_en": "continuous stochastic field",
        "quote": "Niente griglia, niente cella. Solo traiettorie che si aprono come un ventaglio.",
        "quote_en": "No grid, no cell. Only trajectories that open up like a fan.",
        "hashtag": "#stochasticfield #generativelines",
    },
    "06": {
        "nome": "Matrice 06 — Tessuto Vivente",
        "nome_en": "Matrix 06 — Living Tissue",
        "effetto": "Living Automaton Bloom",
        "processo": "Automa Cellulare 2D Continuo + Fioriture Granulari",
        "processo_en": "Continuous 2D Cellular Automaton + Granular Blooms",
        "motore_tag": "matrice che muta da sé",
        "motore_tag_en": "self-mutating matrix",
        "quote": "Non reagisce e basta: cresce e si dirada anche quando nessuno la tocca.",
        "quote_en": "It doesn't just react: it grows and thins out even when nothing touches it.",
        "hashtag": "#cellularautomaton #generativetexture",
    },
    "07": {
        "nome": "Matrice 07 — Flusso Dati",
        "nome_en": "Matrix 07 — Data Stream",
        "effetto": "Data Stream Glyphs",
        "processo": "Colonne di Caratteri Continue + Clock Stocastico",
        "processo_en": "Continuous Character Columns + Stochastic Clock",
        "motore_tag": "flusso continuo di simboli",
        "motore_tag_en": "continuous symbol stream",
        "quote": "Non forme: solo cifre che cadono, mai davvero ferme.",
        "quote_en": "No shapes: only digits falling, never truly still.",
        "hashtag": "#datastream #generativetypography",
    },
    "08": {
        "nome": "Matrice 08 — Traccia Oscilloscopica",
        "nome_en": "Matrix 08 — Oscilloscope Trace",
        "effetto": "Lissajous Trace",
        "processo": "Curva Parametrica Continua + Scia Fosforescente",
        "processo_en": "Continuous Parametric Curve + Phosphor Trail",
        "motore_tag": "curva continua a persistenza",
        "motore_tag_en": "continuous curve with persistence",
        "quote": "Non ci sono più oggetti: solo una linea che non smette di disegnarsi.",
        "quote_en": "There are no more objects: only a line that never stops drawing itself.",
        "hashtag": "#oscilloscope #lissajous",
    },
    "09": {
        "nome": "Matrice 09 — Rete",
        "nome_en": "Matrix 09 — Network",
        "effetto": "Circuit Pulse",
        "processo": "Topologia Fissa + Propagazione d'Impulso",
        "processo_en": "Fixed Topology + Impulse Propagation",
        "motore_tag": "circuito spento finché non toccato",
        "motore_tag_en": "circuit silent until touched",
        "quote": "La rete non cambia mai forma. Solo la corrente che la attraversa, per un istante.",
        "quote_en": "The network never changes shape. Only the current running through it, for an instant.",
        "hashtag": "#networkgraph #circuitart",
    },
    "10": {
        "nome": "Matrice 10 — Spettro Radiale",
        "nome_en": "Matrix 10 — Radial Spectrum",
        "effetto": "Concentric Band Rings",
        "processo": "Tre Anelli Concentrici Reattivi a Bassi/Medi/Alti",
        "processo_en": "Three Concentric Rings Reactive to Bass/Mid/Treble",
        "motore_tag": "tre bande, sempre in ascolto",
        "motore_tag_en": "three bands, always listening",
        "quote": "Non c'è un colpo che accende tutto. Ci sono tre bande che respirano sempre, ognuna per conto suo.",
        "quote_en": "No single hit lights everything up. Three bands breathe constantly, each on its own.",
        "hashtag": "#radialspectrum #audioreactive",
    },
}

USURA_STATE_PATH = os.path.join(tempfile.gettempdir(), "beatglitch_usura_state.json")


def load_usura_count():
    """Numero totale di generazioni fatte finora col modulo Jeck. Persiste su disco
    finché il container resta attivo — si azzera se Streamlit Cloud riavvia
    l'istanza (redeploy, inattività prolungata): è un limite reale, non nascosto."""
    try:
        with open(USURA_STATE_PATH, "r") as f:
            return int(json.load(f).get("count", 0))
    except Exception:
        return 0


def save_usura_count(count):
    try:
        with open(USURA_STATE_PATH, "w") as f:
            json.dump({"count": int(count)}, f)
    except Exception:
        pass  # se il filesystem non è scrivibile, la sessione prosegue senza persistenza


def usura_level_from_count(count, saturation=50, baseline=2.0):
    """0..1, satura dopo 'saturation' generazioni. Un piccolo scarto di base (baseline)
    fa sì che anche la primissima generazione (count=0) mostri già un accenno di
    degrado — altrimenti a zero il modulo tornerebbe identico a Ikeda puro, confondendo
    chi lo prova per la prima volta prima ancora di sapere che l'usura si accumula."""
    return min(1.0, (count + baseline) / float(saturation))


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
    "Bianco":  (255, 255, 255),
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


def _band_continuous_value(band_data, mode, block_seconds, t):
    """Valore 0..1 CONTINUO (non a scatti per blocco) dell'energia di banda
    all'istante t. A differenza di _band_block_value (voluto a gradini netti per
    il Modulo 02), qui interpola in continuo: con audio reale segue l'inviluppo
    campione per campione, in modalità procedurale sfuma tra i centri dei blocchi
    invece di saltare — necessario per una Matrice 10 che risponde davvero in
    tempo reale, non solo ogni pochi secondi."""
    t = max(0.0, t)
    if mode == "audio":
        env, times = band_data["env"], band_data["times"]
        if len(env) == 0:
            return 0.0
        return float(np.interp(t, times, env))
    values = band_data["blocks"]["values"]
    bs = band_data["blocks"]["block_seconds"]
    n = len(values)
    if n == 1:
        return float(values[0])
    centers = (np.arange(n) + 0.5) * bs
    return float(np.interp(t, centers, values))


def generate_deviation_events(duration, seed, audio_events=None, strong_threshold=0.72,
                               min_gap=1.0, block_seconds=3.0):
    """Eventi di deviazione RARI (Molnár): scattano solo sui colpi audio davvero forti
    (non su ogni battito — altrimenti non sarebbero eccezioni), con una distanza minima
    tra un evento e il successivo. Se l'audio non basta a generarne a sufficienza (o non
    c'è audio), un fallback procedurale sparso ne aggiunge qualcuno, sempre con parsimonia."""
    events = []
    if audio_events:
        last_t = -min_gap
        for e in sorted(audio_events, key=lambda x: x["t"]):
            if e["vel"] >= strong_threshold and (e["t"] - last_t) >= min_gap:
                events.append({"t": e["t"], "source": "audio", "vel": e["vel"],
                                "band": e.get("band", "mid"), "pan": e.get("pan", 0.0)})
                last_t = e["t"]

    rng = np.random.RandomState(seed + 909)
    n_blocks = max(1, int(np.ceil(duration / block_seconds)))
    for i in range(n_blocks):
        if rng.random() < 0.35:  # ~1 possibilità su 3 per blocco: resta raro
            t_dev = i * block_seconds + rng.uniform(0.2, max(0.3, block_seconds - 0.2))
            if t_dev >= duration:
                continue
            if not events or min(abs(t_dev - e["t"]) for e in events) >= min_gap * 0.5:
                events.append({"t": float(t_dev), "source": "ca",
                                "vel": float(rng.uniform(0.5, 1.0)), "band": None, "pan": 0.0})

    events.sort(key=lambda e: e["t"])
    return events


DEVIATION_TYPES = ["colore", "posizione", "rotazione", "dimensione"]


def _deviation_cell_and_type(e, n_cells):
    seed = int((e["t"] * 1000) % 100000)
    rng = np.random.RandomState(seed)
    cell_idx = int(rng.randint(0, n_cells))
    dtype = DEVIATION_TYPES[int(rng.randint(0, len(DEVIATION_TYPES)))]
    return cell_idx, dtype


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


def _band_envelopes_over_time(y_mono, sr, n_fft=2048, hop_length=512):
    """Inviluppo di energia continuo per bassi/medi/alti nel tempo (non per singolo
    onset) — usato per far ricomporre il mosaico Henke in base al contenuto reale."""
    import librosa
    stft = np.abs(librosa.stft(y_mono, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)

    def norm(x):
        return x / (x.max() + 1e-9)

    bass = norm(stft[(freqs >= 20) & (freqs < 250)].sum(axis=0))
    mid = norm(stft[(freqs >= 250) & (freqs < 2000)].sum(axis=0))
    treble = norm(stft[(freqs >= 2000) & (freqs < 8000)].sum(axis=0))
    return {"bass": bass, "mid": mid, "treble": treble, "times": times}


def _lowpass_filter(y, sr, cutoff=150.0, order=4):
    """Filtro passa-basso (Butterworth, fase zero) per isolare la cassa/i bassi
    prima di cercare gli attacchi — evita che hi-hat/altri elementi del mix
    mascherino o confondano il rilevatore di colpi."""
    from scipy.signal import butter, filtfilt
    nyq = sr / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, y)


def extract_from_audio(audio_path):
    import librosa
    y_raw, sr = librosa.load(audio_path, sr=SR, mono=False)
    if y_raw.ndim == 1:
        y_stereo = np.stack([y_raw, y_raw])  # mono duplicato: pan sempre 0 (centro)
    else:
        y_stereo = y_raw[:2]
    y_mono = y_stereo.mean(axis=0)
    duration = y_mono.shape[0] / sr

    rms = librosa.feature.rms(y=y_mono)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    band_envelopes = _band_envelopes_over_time(y_mono, sr)
    win = 2048

    # onset "generici" sul mix intero — catturano rullanti/hi-hat/medi-alti
    onset_frames = librosa.onset.onset_detect(y=y_mono, sr=sr, backtrack=True)
    onset_times = list(librosa.frames_to_time(onset_frames, sr=sr))

    # onset DEDICATI alla cassa/bassi: isolati con un passa-basso prima di cercare
    # gli attacchi, così un basso continuo o un hi-hat non nascondono/confondono
    # il colpo — risolve il problema della cassa che "non va a tempo"
    y_bass = _lowpass_filter(y_mono, sr, cutoff=150.0)
    bass_onset_frames = librosa.onset.onset_detect(y=y_bass, sr=sr, backtrack=True,
                                                    pre_max=10, post_max=10, delta=0.15)
    bass_onset_candidates = list(librosa.frames_to_time(bass_onset_frames, sr=sr))

    # verifica incrociata: un candidato è un VERO colpo di cassa solo se, nel mix
    # ORIGINALE non filtrato, i bassi sono davvero la banda dominante in quel punto —
    # scarta i falsi positivi da elementi a banda larga (es. hi-hat) che lasciano un
    # residuo a bassa frequenza anche dopo il filtro, senza essere davvero "bassi"
    bass_onset_times = []
    for bt in bass_onset_candidates:
        sample_pos = int(bt * sr)
        lo, hi = max(0, sample_pos - win // 2), sample_pos + win // 2
        if _dominant_band(y_mono[lo:hi], sr) == "bass":
            bass_onset_times.append(bt)

    # tolgo dagli onset generici quelli troppo vicini a un colpo di cassa già
    # individuato, per non contare due volte lo stesso istante
    onset_times = [ot for ot in onset_times
                    if not any(abs(ot - bt) < 0.06 for bt in bass_onset_times)]

    events = []

    def _make_event(ot, forced_band=None):
        idx = int(np.argmin(np.abs(rms_times - ot)))
        vel = float(min(1.0, rms[idx] / (rms.max() + 1e-9)))
        sample_pos = int(ot * sr)
        lo, hi = max(0, sample_pos - win // 2), sample_pos + win // 2
        band = forced_band or _dominant_band(y_mono[lo:hi], sr)
        bp = BAND_PARAMS[band]
        pan = _pan_of_window(y_stereo[0, lo:hi], y_stereo[1, lo:hi])
        return {"t": float(ot), "dur": bp["dur"], "pitch": bp["pitch"] + int(vel * 8),
                "vel": vel, "source": "audio", "band": band, "pan": pan}

    for bt in bass_onset_times:
        events.append(_make_event(bt, forced_band="bass"))
    for ot in onset_times:
        events.append(_make_event(ot))
    events.sort(key=lambda e: e["t"])

    env = rms / (rms.max() + 1e-9)

    # rilevamento BPM reale — usato da più matrici per calibrare le proprie
    # cadenze sul tempo VERO del brano, invece di costanti fisse in secondi
    try:
        tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
        tempo_val = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size > 0 else 0.0
        bpm = tempo_val if tempo_val > 0 else None
    except Exception:
        bpm = None

    return events, env, duration, y_stereo, band_envelopes, bpm


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
                 video_events=None, video_env=None, audio_band_envelopes=None,
                 resolution=200, macro_block_seconds=3.0, deviation_strong_threshold=0.72,
                 deviation_min_gap=1.0, bpm=None):
    has_external = bool(midi_events or audio_events or video_events)

    # il seed "di base" pilotato dall'utente resta riproducibile, ma viene perturbato
    # dal contenuto reale delle fonti esterne, in modo che due brani diversi diano
    # davvero pattern procedurali diversi, non solo pochi eventi colorati in più
    effective_seed = seed + _content_seed_offset(audio_env, video_env, midi_events)

    macro = generate_macro_envelope_procedural(duration, effective_seed, resolution)
    macro_blocks = generate_macro_blocks(duration, effective_seed, block_seconds=macro_block_seconds)

    # mosaico per banda (modulo Henke): se c'è audio reale, i blocchi bassi/medi/alti
    # riflettono l'energia VERA di quella banda in quel segmento — non rumore casuale.
    # Senza audio, tre sequenze procedurali indipendenti fanno da sostituto plausibile.
    if audio_band_envelopes is not None:
        band_mosaic = {
            "mode": "audio", "block_seconds": macro_block_seconds,
            "bass": {"env": audio_band_envelopes["bass"], "times": audio_band_envelopes["times"]},
            "mid": {"env": audio_band_envelopes["mid"], "times": audio_band_envelopes["times"]},
            "treble": {"env": audio_band_envelopes["treble"], "times": audio_band_envelopes["times"]},
        }
    else:
        band_mosaic = {
            "mode": "procedural", "block_seconds": macro_block_seconds,
            "bass": {"blocks": generate_macro_blocks(duration, effective_seed + 301, macro_block_seconds)},
            "mid": {"blocks": generate_macro_blocks(duration, effective_seed + 302, macro_block_seconds)},
            "treble": {"blocks": generate_macro_blocks(duration, effective_seed + 303, macro_block_seconds)},
        }

    external_envs = [e for e in (audio_env, video_env) if e is not None and len(e) > 1]
    silence_envelope = None  # se presente, pilota il "gate silenzio" nel rendering
    if external_envs:
        resampled = [
            np.interp(np.linspace(0, 1, resolution), np.linspace(0, 1, len(e)), e)
            for e in external_envs
        ]
        combined_ext = np.mean(resampled, axis=0)
        macro = 0.4 * macro + 0.6 * combined_ext  # gli esterni pesano di più se presenti

        # il gate silenzio usa una risoluzione FINE dedicata (indipendente dai 200 punti
        # della matrice): a 200 punti su un brano di 3 minuti ogni campione vale ~0.9s,
        # che si traduce in un ritardo percepibile quando il suono riprende dopo una
        # pausa. Qui usiamo almeno 20 campioni al secondo (~50ms), impercettibile.
        gate_resolution = max(resolution, int(duration * 20))
        gate_resampled = [
            np.interp(np.linspace(0, 1, gate_resolution), np.linspace(0, 1, len(e)), e)
            for e in external_envs
        ]
        silence_envelope = np.mean(gate_resampled, axis=0)

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

    # BPM reale (se rilevato dall'audio) → intervallo di un battito in secondi.
    # Senza rilevamento (nessun audio, o audio troppo atonale per un BPM affidabile)
    # si assume un'ipotesi neutra di 120 BPM, così il comportamento resta coerente
    # anche in modalità puramente procedurale — condiviso da più matrici.
    beat_interval = (60.0 / bpm) if (bpm and bpm > 0) else (60.0 / 120.0)

    # distanza minima tra deviazioni (Molnár) espressa in BATTITI reali, non più
    # secondi assoluti: un brano veloce (es. techno) ha battiti corti → deviazioni
    # più ravvicinate in automatico; uno lento le tiene naturalmente più diradate
    effective_deviation_min_gap = max(0.08, deviation_min_gap * beat_interval)

    deviation_events = generate_deviation_events(
        duration, effective_seed, audio_events=audio_events, block_seconds=macro_block_seconds,
        strong_threshold=deviation_strong_threshold, min_gap=effective_deviation_min_gap,
    )

    # campo automa cellulare 2D dedicato (modulo 06): non serve a generare eventi,
    # è esso stesso il "tessuto" visivo/sonoro che scorre e muta nel tempo — la sua
    # cadenza è agganciata al battito reale (5 "generazioni" per battito), non a un
    # ritmo fisso: un brano veloce fa mutare il tessuto più in fretta, uno lento più
    # lentamente
    steps_per_beat = 5.0
    automaton_steps = max(30, int((duration / beat_interval) * steps_per_beat))
    automaton_field = cellular_automaton(rule, width=140, steps=automaton_steps, seed=effective_seed + 909)

    return {
        "duration": duration,
        "seed": effective_seed,
        "macro_blocks": macro_blocks,
        "band_mosaic": band_mosaic,
        "deviation_events": deviation_events,
        "events": events,
        "macro_envelope": macro,
        "silence_envelope": silence_envelope,
        "micro_texture": texture,
        "automaton_field": automaton_field,
        "bpm": bpm,
        "beat_interval": beat_interval,
    }


# ============================================================
# 4. GENERATORE VISIVO
# ============================================================

def _lane_fraction(e, position_mode="pan"):
    """Posizione 0..1 lungo l'asse delle corsie.
    - 'pan': usa il pan stereo reale (eventi audio) o il pitch come fallback. Se il
      mix è centrato/mono, tutti i colpi finiscono vicino al centro indipendentemente
      dal numero di corsie impostato — è un riflesso del mix reale, non un bug.
    - 'frequenza': ignora il pan e usa la banda (bassi=sinistra, alti=destra),
      garantendo sempre distribuzione sull'intera larghezza qualunque sia il mix."""
    if position_mode == "frequenza":
        band = e.get("band")
        band_center = {"bass": 0.15, "mid": 0.5, "treble": 0.85}.get(band, 0.5)
        fine = ((e.get("pitch", 60) % 12) / 12 - 0.5) * 0.25
        return min(0.999, max(0.0, band_center + fine))
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
                        num_lanes=10, palette="Multicolore (per fonte)", band_colors=None,
                        position_mode="pan"):
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

        lane_frac = min(0.999, max(0.0, _lane_fraction(e, position_mode)))
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


def _band_block_value(band_data, mode, block_seconds, t):
    """Valore 0..1 del blocco corrente per una banda: media dell'energia reale
    dell'audio in quel segmento (modo 'audio'), o sequenza procedurale di fallback."""
    t = max(0.0, t)
    if mode == "audio":
        env, times = band_data["env"], band_data["times"]
        if len(env) == 0:
            return 0.0
        block_start = (t // block_seconds) * block_seconds
        block_end = block_start + block_seconds
        mask = (times >= block_start) & (times < block_end)
        if not np.any(mask):
            idx = min(len(env) - 1, int(np.searchsorted(times, t)))
            return float(env[idx])
        return float(env[mask].mean())
    return macro_block_value(band_data["blocks"], t)


def render_frame_henke(t, score, width=960, height=540, orientation="verticale",
                        num_lanes=10, palette="Multicolore (per fonte)", band_colors=None):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    res = len(score["macro_envelope"])
    env_idx = min(res - 1, int((t / max(score["duration"], 1e-6)) * res))
    macro_v = float(score["macro_envelope"][env_idx])

    grid_alpha = int(12 + macro_v * 18)
    grid_step = max(8, width // 24)
    for gx in range(0, width, grid_step):
        frame[:, gx] = (grid_alpha, grid_alpha, grid_alpha)

    # MOSAICO (Henke): 3 righe — alti/medi/bassi dall'alto in basso — 4 celle ciascuna.
    # Il livello base (lento, ~3s) resta uguale su tutta la riga: nessuno sfasamento
    # meccanico artificiale, quello dava l'impressione di "accendersi in sequenza"
    # a prescindere dal brano. Il ritmo vero arriva dai colpi reali (sotto).
    mosaic = score["band_mosaic"]
    mode, block_seconds = mosaic["mode"], mosaic["block_seconds"]
    bc = band_colors or DEFAULT_BAND_COLORS
    rows_bands = ["treble", "mid", "bass"]
    cols = 4
    cell_w, cell_h = width / cols, height / len(rows_bands)

    base_vals = {}
    for r, band_name in enumerate(rows_bands):
        val = _band_block_value(mosaic[band_name], mode, block_seconds, t)
        base_vals[band_name] = val
        fill = 0.22 + 0.55 * val
        shade = 0.30 + 0.55 * val
        base_color = bc.get(band_name, (140, 140, 150))
        color = tuple(int(ch * shade) for ch in base_color)
        for c in range(cols):
            bw, bh = cell_w * fill, cell_h * fill
            cx, cy = c * cell_w + cell_w / 2, r * cell_h + cell_h / 2
            x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
            x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)

    # GUIZZO RITMICO: ogni colpo reale (cassa/rullante/hi-hat, dagli onset audio)
    # accende brevemente la cella della sua riga — la colonna dipende dal pan reale
    # del colpo, quindi si sposta leggermente invece di essere sempre nello stesso punto.
    for e in score["events"]:
        band_name = e.get("band")
        if e.get("source") != "audio" or band_name not in rows_bands:
            continue
        pulse_dur = max(e["dur"], 0.12)
        if not (e["t"] <= t <= e["t"] + pulse_dur):
            continue
        prog = (t - e["t"]) / pulse_dur
        decay = 1.0 - prog  # il guizzo si spegne nel corso della sua durata

        r = rows_bands.index(band_name)
        lane_frac = min(0.999, max(0.0, _lane_fraction(e)))
        c = int(lane_frac * cols)

        pulse_fill = min(0.98, (0.22 + 0.55 * base_vals[band_name]) + 0.45 * decay * e["vel"])
        pulse_shade = min(1.0, 0.6 + 0.4 * decay * e["vel"])
        base_color = bc.get(band_name, (140, 140, 150))
        color = tuple(int(ch * pulse_shade) for ch in base_color)

        bw, bh = cell_w * pulse_fill, cell_h * pulse_fill
        cx, cy = c * cell_w + cell_w / 2, r * cell_h + cell_h / 2
        x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
        x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)

    return frame


def render_frame_molnar(t, score, width=960, height=540, orientation="verticale",
                         num_lanes=10, palette="Multicolore (per fonte)", band_colors=None,
                         grid_cols=10, accent_color=(235, 40, 60)):
    """Griglia rigida (Molnár): quasi sempre perfettamente regolare. Le celle non
    respirano, non reagiscono all'istante — restano identiche finché non arriva una
    deviazione RARA (colore/posizione/rotazione/dimensione), sempre sulla stessa
    griglia di partenza. Nessuna texture di sfondo: il fondo è vuoto apposta."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    cols = max(3, int(grid_cols))
    cell_size = width / cols
    rows = max(1, int(round(height / cell_size)))
    cell_w, cell_h = width / cols, height / rows
    base_color = (150, 150, 150)
    base_square = min(cell_w, cell_h) * 0.34

    # griglia regolare — stessa dimensione, stesso colore, stesso orientamento ovunque
    cell_state = {}  # cell_idx -> parametri di deviazione attivi in questo frame
    for e in score["deviation_events"]:
        dur = 0.45
        if not (e["t"] <= t <= e["t"] + dur):
            continue
        n_cells = cols * rows
        cell_idx, dtype = _deviation_cell_and_type(e, n_cells)
        prog = (t - e["t"]) / dur
        decay = 1.0 - prog
        cell_state[cell_idx] = (dtype, decay, e)

    for idx in range(cols * rows):
        r, c = divmod(idx, cols)
        cx, cy = c * cell_w + cell_w / 2, r * cell_h + cell_h / 2
        half = base_square

        if idx in cell_state:
            dtype, decay, e = cell_state[idx]
            # colore diverso per banda (come nelle altre matrici): un colpo di
            # bassi devia in un colore, uno di alti in un altro. Le deviazioni
            # procedurali (senza banda, quando non c'è audio) usano un grigio chiaro.
            bc = band_colors or DEFAULT_BAND_COLORS
            band_name = e.get("band")
            target_color = bc.get(band_name, accent_color) if band_name else accent_color
            color = tuple(int(a + (b - a) * decay) for a, b in zip(base_color, target_color))
            offset, angle, half_eff = (0, 0), 0.0, half

            if dtype == "posizione":
                shift = half * 3.4 * decay  # movimento più ampio, prima era appena percettibile
                offset = (shift, -shift * 0.6)
            elif dtype == "dimensione":
                half_eff = half * (1.0 + 2.2 * decay)
            elif dtype == "rotazione":
                angle = 45.0 * decay
            # "colore": solo il lampeggio, nessuna geometria aggiuntiva
        else:
            color, half_eff, offset, angle = base_color, half, (0, 0), 0.0

        px, py = cx + offset[0], cy + offset[1]
        if abs(angle) > 1.0:
            rect = ((px, py), (half_eff * 2, half_eff * 2), angle)
            box = cv2.boxPoints(rect).astype(int)
            cv2.fillConvexPoly(frame, box, color)
        else:
            x0, y0 = int(px - half_eff), int(py - half_eff)
            x1, y1 = int(px + half_eff), int(py + half_eff)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)

    return frame


def apply_visual_degradation(frame, t, seed, usura_level, memory_frame=None):
    """Strisce DISTRUTTE (Jeck): non un velo di rumore sopra l'immagine pulita, ma
    tagli orizzontali che si spostano, canali RGB sfalsati (aberrazione cromatica),
    inversioni, e — quando disponibile — un fotogramma passato che riaffiora a
    strisce dentro quello presente (la "memoria" del nastro che sanguina).
    Anche la primissima generazione (usura minima) deve mostrarsi chiaramente
    diversa da Ikeda: la probabilità parte già alta e sale ulteriormente con l'uso."""
    if usura_level <= 0.0:
        return frame
    h, w = frame.shape[:2]
    out = frame.copy()

    n_strips = 16
    strip_h = max(1, h // n_strips)
    rng = np.random.RandomState(int((t * 97 + seed * 11) % (2**31)))

    trigger_prob = min(0.75, 0.22 + 0.6 * usura_level)  # base visibile subito, cresce con l'uso
    shift_frac = min(0.5, 0.15 + 0.4 * usura_level)     # spostamento minimo garantito
    channel_shift_max = 20 + int(50 * usura_level)

    for s in range(n_strips):
        y0 = s * strip_h
        y1 = h if s == n_strips - 1 else y0 + strip_h
        if rng.random() >= trigger_prob:
            continue
        effetto = rng.choice(["spostamento", "canale", "inversione", "memoria"])

        if effetto == "spostamento":
            max_shift = int(w * shift_frac) + 1
            shift = rng.randint(-max_shift, max_shift + 1)
            out[y0:y1] = np.roll(out[y0:y1], shift, axis=1)
        elif effetto == "canale":
            c = rng.randint(0, 3)
            cshift = rng.randint(10, max(11, channel_shift_max))
            out[y0:y1, :, c] = np.roll(out[y0:y1, :, c], cshift, axis=1)
        elif effetto == "inversione":
            out[y0:y1] = 255 - out[y0:y1]
        elif effetto == "memoria" and memory_frame is not None:
            out[y0:y1] = memory_frame[y0:y1]

    noise_amp = 6 + 18 * usura_level  # grana visibile fin da subito, non solo ad usura alta
    noise = rng.normal(0, noise_amp, out.shape)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return out


def render_frame_jeck(t, score, width=960, height=540, orientation="verticale",
                       num_lanes=10, palette="Multicolore (per fonte)", band_colors=None,
                       usura_level=0.0):
    """Riusa la resa 'pulita' di Ikeda come base — lo stesso dato, la stessa
    grammatica visiva — e la distrugge in strisce in base a quante volte è già
    stata suonata: più usura, più tagli, spostamenti e memoria che riaffiora."""
    base = render_frame_ikeda(t, score, width=width, height=height, orientation=orientation,
                               num_lanes=num_lanes, palette=palette, band_colors=band_colors)

    memory_frame = None
    if usura_level > 0.0:
        memory_delay = 0.4 + 1.5 * usura_level  # più usura, più "vecchio" il ricordo che riaffiora
        memory_frame = render_frame_ikeda(max(0.0, t - memory_delay), score, width=width, height=height,
                                           orientation=orientation, num_lanes=num_lanes,
                                           palette=palette, band_colors=band_colors)

    return apply_visual_degradation(base, t, score["seed"], usura_level, memory_frame=memory_frame)


def render_frame_stocastico(t, score, width=960, height=540, orientation="verticale",
                             num_lanes=10, palette="Multicolore (per fonte)", band_colors=None,
                             position_mode="pan"):
    """Matrice 05: campo continuo stocastico. Niente griglia, niente celle
    rettangolari: ogni evento genera un FASCIO di linee diagonali che attraversa
    l'intero fotogramma da un bordo all'altro, con partenza e arrivo che divergono
    in modo casuale (le linee non sono mai parallele, si aprono come un ventaglio).
    Lo sfondo è un pulviscolo di punti sparsi a densità variabile, non una griglia.
    Stesso sistema dati/colori delle altre matrici (get_event_color, band_colors,
    palette, gate silenzio), ma grammatica visiva radicalmente diversa: continua
    e obliqua invece che discreta e ortogonale."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    res = len(score["macro_envelope"])
    env_idx = min(res - 1, int((t / max(score["duration"], 1e-6)) * res))
    macro_v = float(score["macro_envelope"][env_idx])

    # pulviscolo di fondo: punti sparsi con seed legato al tempo (non una griglia fissa)
    rng_bg = np.random.RandomState(int((t * 30) % (2 ** 31)))
    n_bg_points = int(40 + macro_v * 90)
    bg_alpha = int(10 + macro_v * 14)
    xs = rng_bg.uniform(0, width - 1, n_bg_points)
    ys = rng_bg.uniform(0, height - 1, n_bg_points)
    for x, y in zip(xs.astype(int), ys.astype(int)):
        frame[y, x] = (bg_alpha, bg_alpha, bg_alpha)

    SILENCE_THRESHOLD = 0.06
    gate_env = score.get("silence_envelope")
    if gate_env is not None:
        gate_idx = min(len(gate_env) - 1, int((t / max(score["duration"], 1e-6)) * len(gate_env)))
        if float(gate_env[gate_idx]) < SILENCE_THRESHOLD:
            return frame

    active = [e for e in score["events"] if e["t"] <= t <= e["t"] + max(e["dur"], 0.05)]
    num_lanes = max(1, int(num_lanes))

    for e in active:
        prog = (t - e["t"]) / max(e["dur"], 0.05)
        color = get_event_color(e, palette, band_colors)
        fade = 1.0 - prog * 0.6

        seed_e = int((e["t"] * 1000) % (2 ** 31))
        rng_e = np.random.RandomState(seed_e)

        lane_frac = min(0.999, max(0.0, _lane_fraction(e, position_mode)))
        band_thickness = BAND_PARAMS[e["band"]]["thickness"] if "band" in e else \
            (1.0 if e["source"] == "ca" else 1.6)

        # numero di linee del fascio e dispersione: proporzionali a num_lanes/intensità
        n_lines = max(2, int(num_lanes * (0.25 + 0.65 * e["vel"])))
        spread_start = 0.05 + 0.08 * e["vel"]
        spread_end = spread_start * (1.8 + prog)  # il fascio diverge nel tempo

        event_orientation = _event_orientation(e, orientation)
        thickness = max(1, int(round(1 + band_thickness)))
        c = tuple(int(ch * fade) for ch in color)

        for _ in range(n_lines):
            start_frac = min(0.999, max(0.0, lane_frac + rng_e.uniform(-spread_start, spread_start)))
            end_frac = min(0.999, max(0.0, lane_frac + rng_e.uniform(-spread_end, spread_end)))

            if event_orientation == "verticale":
                x0, x1 = int(start_frac * width), int(end_frac * width)
                y0, y1 = 0, height - 1
            else:
                y0, y1 = int(start_frac * height), int(end_frac * height)
                x0, x1 = 0, width - 1

            cv2.line(frame, (x0, y0), (x1, y1), c, thickness, lineType=cv2.LINE_AA)

    return frame


def render_frame_automaton(t, score, width=960, height=540, orientation="verticale",
                            num_lanes=10, palette="Multicolore (per fonte)", band_colors=None,
                            position_mode="pan"):
    """Matrice 06: qui non è un evento a disegnare una forma su uno sfondo statico —
    è la MATRICE STESSA a vivere e mutare nel tempo. Un automa cellulare 2D dedicato
    scorre come tessuto organico continuo sullo sfondo (mai rigido, mai a celle
    separate). Gli eventi audio non tracciano barre né linee ma fioriture di colore
    morbide (gradienti radiali concentrici) che sbocciano sopra il tessuto — nessuna
    forma geometrica netta, nessuna griglia: la texture di fondo è essa stessa il
    contenuto generativo, non solo una cornice per gli eventi."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    duration = max(score["duration"], 1e-6)

    field = score["automaton_field"]
    steps, cols = field.shape
    row_idx = min(steps - 1, int((t / duration) * steps))
    window = 90  # "storia" visibile dell'automa, come un tessuto che scorre nel tempo
    start_row = max(0, row_idx - window + 1)
    strip = field[start_row:row_idx + 1]
    if strip.shape[0] < window:
        pad = np.zeros((window - strip.shape[0], cols), dtype=field.dtype)
        strip = np.vstack([pad, strip])

    tex_img = cv2.resize((strip.astype(np.float32) * 255.0), (width, height),
                          interpolation=cv2.INTER_LINEAR)
    tex_img = cv2.GaussianBlur(tex_img, (0, 0), sigmaX=3)

    res = len(score["macro_envelope"])
    env_idx = min(res - 1, int((t / duration) * res))
    macro_v = float(score["macro_envelope"][env_idx])
    tex_level = 14 + macro_v * 22
    tex_gray = np.clip(tex_img / 255.0 * tex_level, 0, 255).astype(np.uint8)
    frame[:, :, 0] = tex_gray
    frame[:, :, 1] = tex_gray
    frame[:, :, 2] = tex_gray

    SILENCE_THRESHOLD = 0.06
    gate_env = score.get("silence_envelope")
    if gate_env is not None:
        gate_idx = min(len(gate_env) - 1, int((t / duration) * len(gate_env)))
        if float(gate_env[gate_idx]) < SILENCE_THRESHOLD:
            return frame

    active = [e for e in score["events"] if e["t"] <= t <= e["t"] + max(e["dur"], 0.05)]
    num_lanes = max(1, int(num_lanes))

    for e in active:
        prog = (t - e["t"]) / max(e["dur"], 0.05)
        color = get_event_color(e, palette, band_colors)
        fade = 1.0 - prog * 0.5

        lane_frac = min(0.999, max(0.0, _lane_fraction(e, position_mode)))
        band_thickness = BAND_PARAMS[e["band"]]["thickness"] if "band" in e else \
            (1.0 if e["source"] == "ca" else 1.6)
        event_orientation = _event_orientation(e, orientation)

        radius = max(4, int((22 + 55 * e["vel"]) * band_thickness * (1 - prog * 0.3) *
                             (0.4 + 0.6 * (num_lanes / 10.0))))

        if event_orientation == "verticale":
            cx, cy = int(lane_frac * width), height // 2
        else:
            cx, cy = width // 2, int(lane_frac * height)

        n_rings = 6
        for ring in range(n_rings, 0, -1):
            r = max(1, int(radius * ring / n_rings))
            alpha = fade * (0.15 + 0.85 * (1 - ring / n_rings))
            c = tuple(int(ch * alpha) for ch in color)
            cv2.circle(frame, (cx, cy), r, c, -1, lineType=cv2.LINE_AA)

    return frame


def _datastream_draw_column(frame, col, col_w, n_rows, char_h, t, scroll_speed_base, glyphs,
                             font, font_scale, base_gray, lane_color, height, cycle_len=9):
    # velocità e fase PROPRIE della colonna (stabili nel tempo, diverse da colonna
    # a colonna): è questo che rompe l'effetto "blocco che si muove all'unisono" —
    # ogni striscia di dati cade per conto suo, come nel riferimento reale
    rng_speed = np.random.RandomState((col * 7793 + 12345) & 0x7FFFFFFF)
    speed_mult = 0.55 + rng_speed.random() * 0.9
    phase0 = rng_speed.random()
    own_speed = scroll_speed_base * speed_mult
    offset = (t * own_speed + phase0) % 1.0
    time_step = int(t * own_speed + phase0 * 1000)

    seed_c = (col * 9973 + time_step) & 0x7FFFFFFF
    rng_c = np.random.RandomState(seed_c)
    picks = rng_c.random(n_rows)
    char_idx = rng_c.randint(0, len(glyphs), n_rows)
    color_pick = rng_c.random(n_rows)

    # colore già sfumato in base alla distanza dalla corsia attiva più vicina —
    # ogni colonna ha la SUA intensità, colonna per colonna, non a gruppi
    if lane_color is not None:
        color_full, strength = lane_color
        blended = tuple(base_gray + (cf - base_gray) * strength for cf in color_full)
    else:
        blended = None

    x = int(col * col_w + col_w * 0.25)
    for row in range(n_rows):
        if picks[row] > 0.95:
            continue
        y = int((row - offset) * char_h)
        if y < char_h or y > height:
            continue
        # onda di luminosità che scorre con la stessa fase dello scroll: dà un
        # capofila più acceso seguito da una scia che si affievolisce, invece di
        # glifi tutti alla stessa intensità — è questo che fa "sentire" la caduta
        wave = 0.15 + 0.85 * (0.5 + 0.5 * np.cos(2 * np.pi * (row - offset) / cycle_len))
        base_color = blended if (blended is not None and color_pick[row] < 0.85) else \
            (base_gray, base_gray, base_gray)
        color = tuple(min(255, max(0, int(ch * wave * 1.4))) for ch in base_color)
        cv2.putText(frame, glyphs[char_idx[row]], (x, y), font, font_scale, color, 2, cv2.LINE_AA)


def _datastream_draw_row(frame, lane, row_h, n_cols, char_h, t, scroll_speed_base, glyphs,
                          font, font_scale, base_gray, lane_color, width, cycle_len=9):
    rng_speed = np.random.RandomState((lane * 7793 + 54321) & 0x7FFFFFFF)
    speed_mult = 0.55 + rng_speed.random() * 0.9
    phase0 = rng_speed.random()
    own_speed = scroll_speed_base * speed_mult
    offset = (t * own_speed + phase0) % 1.0
    time_step = int(t * own_speed + phase0 * 1000)

    seed_c = (lane * 9973 + time_step) & 0x7FFFFFFF
    rng_c = np.random.RandomState(seed_c)
    picks = rng_c.random(n_cols)
    char_idx = rng_c.randint(0, len(glyphs), n_cols)
    color_pick = rng_c.random(n_cols)

    if lane_color is not None:
        color_full, strength = lane_color
        blended = tuple(base_gray + (cf - base_gray) * strength for cf in color_full)
    else:
        blended = None

    y = int(lane * row_h + row_h * 0.6)
    for col in range(n_cols):
        if picks[col] > 0.95:
            continue
        x = int((col - offset) * char_h)
        if x < char_h or x > width:
            continue
        wave = 0.15 + 0.85 * (0.5 + 0.5 * np.cos(2 * np.pi * (col - offset) / cycle_len))
        base_color = blended if (blended is not None and color_pick[col] < 0.85) else \
            (base_gray, base_gray, base_gray)
        color = tuple(min(255, max(0, int(ch * wave * 1.4))) for ch in base_color)
        cv2.putText(frame, glyphs[char_idx[col]], (x, y), font, font_scale, color, 2, cv2.LINE_AA)


def render_frame_datastream(t, score, width=960, height=540, orientation="verticale",
                             num_lanes=10, palette="Multicolore (per fonte)", band_colors=None,
                             position_mode="pan"):
    """Matrice 07: nessuna forma geometrica — una GRIGLIA DENSA di caratteri
    monospace riempie tutta la larghezza/altezza (nessun vuoto tra le colonne),
    come un readout che non si ferma mai. Ogni STRISCIA (gruppo di colonne/righe
    dense contigue, una per 'corsia' audio-reattiva) scorre da sé; quando un
    evento è attivo nella sua corsia, l'intera striscia — non un solo carattere
    — si accende col colore dell'evento. In modalità 'misto' strisce verticali
    e orizzontali convivono nello stesso fotogramma. Nessuna barra, nessun
    blocco: solo simboli che cadono, fitti, a coprire tutta la scena."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    duration = max(score["duration"], 1e-6)
    res = len(score["macro_envelope"])
    env_idx = min(res - 1, int((t / duration) * res))
    macro_v = float(score["macro_envelope"][env_idx])

    SILENCE_THRESHOLD = 0.06
    gate_env = score.get("silence_envelope")
    silent = False
    if gate_env is not None:
        gate_idx = min(len(gate_env) - 1, int((t / duration) * len(gate_env)))
        silent = float(gate_env[gate_idx]) < SILENCE_THRESHOLD

    num_lanes = max(1, int(num_lanes))
    glyphs = "0123456789ABCDEFXZNMTYJHKLQR#%&@*+=<>"
    char_h = max(10, height // 28)
    char_w = max(6, int(char_h * 0.62))  # spaziatura fitta: colonne/righe dense, senza vuoti
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, char_h / 28.0)
    base_gray = int(28 + macro_v * 18)

    # scia persistente: non solo l'istante esatto dell'evento, ma anche un
    # breve strascico dopo — altrimenti con pochi colpi simultanei si vede un
    # solo guizzo isolato alla volta invece di un flusso vivo e pieno
    trail_window = max(0.35, (score.get("beat_interval") or 0.5) * 2.0)
    active_events_list = []
    if not silent:
        for e in score["events"]:
            e_end = e["t"] + max(e["dur"], 0.05)
            if e["t"] <= t <= e_end:
                recency = 1.0
            elif e_end < t <= e_end + trail_window:
                recency = max(0.0, 1.0 - (t - e_end) / trail_window)
            else:
                continue
            lane_frac = min(0.999, max(0.0, _lane_fraction(e, position_mode)))
            active_events_list.append((e, lane_frac, recency))

    spread_cols = 1.4  # colonne isolate ma non puntiformi: 3-5 colonne per guizzo

    def _make_lane_color_fn(n_positions):
        """Restituisce una funzione pos -> (colore, intensità) sfumati per distanza
        IN COLONNE dalla posizione dell'evento attivo/recente più vicino — non più
        per distanza di 'corsia' (che raggruppava molte colonne insieme). Ogni
        singola colonna/riga densa reagisce per conto proprio; la scia (recency)
        fa sì che più eventi restino visibili insieme invece che un guizzo isolato
        alla volta."""
        if not active_events_list:
            return lambda pos: None
        event_positions = [(get_event_color(e, palette, band_colors), frac * n_positions, recency)
                            for e, frac, recency in active_events_list]

        def _fn(pos):
            accum = np.zeros(3, dtype=float)
            total_w = 0.0
            for color, epos, recency in event_positions:
                dist = abs(pos - epos)
                w = np.exp(-dist / spread_cols) * recency
                if w < 0.12:
                    continue
                accum += np.array(color, dtype=float) * w
                total_w += w
            if total_w <= 0:
                return None
            return tuple(accum / total_w), min(1.0, total_w)
        return _fn

    # velocità di scorrimento agganciata al battito reale: 2 celle per battito come
    # base, invece di una costante arbitraria — un brano veloce fa scorrere il
    # flusso più in fretta, uno lento più adagio; l'energia del brano aggiunge
    # comunque una modulazione sopra questa base
    beat_interval = score.get("beat_interval") or 0.5
    scroll_speed = (2.0 / beat_interval) * (0.7 + 0.6 * macro_v)
    n_rows = int(height / char_h) + 2
    n_cols = int(width / char_h) + 2

    n_cols_dense = max(num_lanes, int(width / char_w))
    n_rows_dense = max(num_lanes, int(height / char_w))
    col_span = max(1, n_cols_dense // num_lanes)
    row_span = max(1, n_rows_dense // num_lanes)

    if orientation == "verticale":
        color_fn = _make_lane_color_fn(n_cols_dense)
        for col in range(n_cols_dense):
            lane_color = color_fn(col)
            _datastream_draw_column(frame, col, char_w, n_rows, char_h, t, scroll_speed, glyphs,
                                     font, font_scale, base_gray, lane_color, height)
    elif orientation == "orizzontale":
        color_fn = _make_lane_color_fn(n_rows_dense)
        for row in range(n_rows_dense):
            lane_color = color_fn(row)
            _datastream_draw_row(frame, row, char_w, n_cols, char_h, t, scroll_speed, glyphs,
                                  font, font_scale, base_gray, lane_color, width)
    else:  # misto: strisce verticali e orizzontali convivono nello stesso fotogramma
        color_fn_v = _make_lane_color_fn(n_cols_dense)
        color_fn_h = _make_lane_color_fn(n_rows_dense)
        for col in range(n_cols_dense):
            lane_idx = min(num_lanes - 1, col // col_span)
            if lane_idx % 2 != 0:
                continue
            lane_color = color_fn_v(col)
            _datastream_draw_column(frame, col, char_w, n_rows, char_h, t, scroll_speed, glyphs,
                                     font, font_scale, base_gray, lane_color, height)
        for row in range(n_rows_dense):
            lane_idx = min(num_lanes - 1, row // row_span)
            if lane_idx % 2 == 0:
                continue
            lane_color = color_fn_h(row)
            _datastream_draw_row(frame, row, char_w, n_cols, char_h, t, scroll_speed, glyphs,
                                  font, font_scale, base_gray, lane_color, width)

    return frame


def render_frame_oscilloscopio(t, score, width=960, height=540, orientation="verticale",
                                num_lanes=10, palette="Multicolore (per fonte)", band_colors=None):
    """Matrice 08: nessun oggetto discreto — una SOLA TRACCIA CONTINUA disegna una
    figura di Lissajous, come l'oscilloscopio di un synth analogico. Non ci sono
    eventi che generano forme proprie: tutto è una curva parametrica unica che si
    deforma nel tempo, con una scia fosforescente (persistenza) che sfuma invece
    di essere ridisegnata da zero. 'Numero di linee' qui regola la risoluzione/
    lunghezza della scia, non un numero di corsie."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    duration = max(score["duration"], 1e-6)
    res = len(score["macro_envelope"])
    env_idx = min(res - 1, int((t / duration) * res))
    macro_v = float(score["macro_envelope"][env_idx])

    cx, cy = width / 2.0, height / 2.0
    if orientation == "verticale":
        amp_x, amp_y = width * 0.28, height * 0.40
    elif orientation == "orizzontale":
        amp_x, amp_y = width * 0.40, height * 0.28
    else:
        amp_x, amp_y = width * 0.36, height * 0.36

    seed = score["seed"]
    freq_x = 3 + (seed % 5)
    freq_y = 2 + ((seed // 5) % 5)
    phase_shift = t * (0.03 + 0.05 * macro_v)

    # frequenza base agganciata al battito reale: un ciclo completo dell'unità di
    # base corrisponde a una battuta intera (4 battiti), invece di una costante
    # arbitraria — freq_x/freq_y restano i rapporti armonici che danno la forma
    # della figura, ma il suo "passo" ora è quello vero del brano
    beat_interval = score.get("beat_interval") or 0.5
    base_rate = 1.0 / (beat_interval * 4.0)

    n_lanes = max(1, int(num_lanes))
    n_trail = max(24, n_lanes * 12)
    trail_span = 0.6
    amp_scale = 0.5 + 0.5 * macro_v

    # eventi che potrebbero toccare la finestra della scia (t - trail_span .. t)
    window_events = [e for e in score["events"]
                     if e["t"] <= t and e["t"] + max(e["dur"], 0.05) >= t - trail_span]

    for i in range(n_trail):
        frac = i / n_trail
        tt = t - frac * trail_span
        if tt < 0:
            continue
        ax, ay = amp_x * amp_scale, amp_y * amp_scale
        x = cx + ax * np.sin(2 * np.pi * freq_x * tt * base_rate + phase_shift)
        y = cy + ay * np.sin(2 * np.pi * freq_y * tt * base_rate + phase_shift * 1.3 + 0.7)
        alpha = (1.0 - frac) ** 2

        seg_color = None
        for e in window_events:
            if e["t"] <= tt <= e["t"] + max(e["dur"], 0.05):
                seg_color = get_event_color(e, palette, band_colors)
                break

        if seg_color is not None:
            color = tuple(int(c * alpha) for c in seg_color)
        else:
            g = int(130 * alpha)
            color = (g, g, g)

        r = max(2, int(3 + 4 * alpha))
        cv2.circle(frame, (int(x), int(y)), r, color, -1, lineType=cv2.LINE_AA)

    return frame


@lru_cache(maxsize=64)
def _network_topology(seed, n_nodes):
    """Topologia FISSA (nodi + archi) per la Matrice 09 — a differenza di tutte le
    altre matrici, qui non nasce/scorre/decade nulla nel tempo: la rete è statica,
    calcolata una volta sola dal seed. Solo l'ACCENSIONE dei nodi/archi cambia
    nel tempo, in risposta agli eventi."""
    n_nodes = max(3, int(n_nodes))
    rng = np.random.RandomState((int(seed) * 1013 + n_nodes * 7919) & 0x7FFFFFFF)
    positions = rng.uniform(0.03, 0.97, size=(n_nodes, 2))  # quasi bordo a bordo, come un mesh che riempie tutta la scena

    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    k = 5  # più collegamenti per nodo = mesh triangolato più fitto quando si accende
    order = np.argsort(dist, axis=1)
    edges = set()
    for i in range(n_nodes):
        for j in order[i, :k]:
            edges.add(tuple(sorted((int(i), int(j)))))

    neighbors = {i: [] for i in range(n_nodes)}
    for (a, b) in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)

    return positions, sorted(edges), neighbors


def render_frame_rete(t, score, width=960, height=540, orientation="verticale",
                       num_lanes=10, palette="Multicolore (per fonte)", band_colors=None,
                       position_mode="pan"):
    """Matrice 09: l'unica con una TOPOLOGIA FISSA — nodi e collegamenti non
    nascono, non scorrono, non decadono: sono sempre gli stessi per tutta la
    durata, e restano completamente INVISIBILI finché nessun evento li tocca (il
    fotogramma è nero puro fuori dagli eventi). Un evento non disegna un oggetto
    proprio: accende il nodo più vicino alla sua posizione e l'impulso si propaga
    per un istante lungo i collegamenti vicini, come corrente in un circuito —
    poi si spegne e torna tutto nero. 'Numero di linee' qui è il numero di nodi
    della rete."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    duration = max(score["duration"], 1e-6)

    n_nodes = max(3, int(num_lanes) * 4)
    positions, edges, neighbors = _network_topology(int(score["seed"]), n_nodes)
    px = (positions[:, 0] * width).astype(int)
    py = (positions[:, 1] * height).astype(int)

    # rete spenta completamente invisibile: nessun disegno di nodi/archi statici —
    # si vede solo ciò che si accende a tempo con la musica

    SILENCE_THRESHOLD = 0.06
    gate_env = score.get("silence_envelope")
    if gate_env is not None:
        gate_idx = min(len(gate_env) - 1, int((t / duration) * len(gate_env)))
        if float(gate_env[gate_idx]) < SILENCE_THRESHOLD:
            return frame

    band_y = {"bass": 0.85, "mid": 0.5, "treble": 0.15}
    active = [e for e in score["events"] if e["t"] <= t <= e["t"] + max(e["dur"], 0.05)]

    for e in active:
        prog = (t - e["t"]) / max(e["dur"], 0.05)
        fade = 1.0 - prog
        color = get_event_color(e, palette, band_colors)

        ex = min(0.999, max(0.0, _lane_fraction(e, position_mode)))
        ey = band_y.get(e.get("band"), 0.5)
        if orientation == "orizzontale":
            ex, ey = ey, ex  # scambia gli assi: la corsia diventa verticale invece che orizzontale

        d = (positions[:, 0] - ex) ** 2 + (positions[:, 1] - ey) ** 2
        origin = int(np.argmin(d))

        origin_c = tuple(int(c * fade) for c in color)
        r_origin = max(7, int(9 + 12 * e["vel"] * fade))
        cv2.circle(frame, (px[origin], py[origin]), r_origin, origin_c, -1, cv2.LINE_AA)
        # alone esterno: rende il nodo acceso riconoscibile a colpo d'occhio, non
        # confondibile con i nodi spenti della rete
        halo_c = tuple(int(c * fade * 0.5) for c in color)
        cv2.circle(frame, (px[origin], py[origin]), r_origin + 4, halo_c, 2, cv2.LINE_AA)

        # propagazione dell'impulso per più salti lungo la rete, con decadimento —
        # con molti eventi attivi insieme, la parte illuminata copre più scena
        visited = {origin: 0}
        frontier = [origin]
        max_hops = 4
        for hop in range(1, max_hops + 1):
            next_frontier = []
            for n in frontier:
                for nb in neighbors.get(n, []):
                    if nb not in visited:
                        visited[nb] = hop
                        next_frontier.append(nb)
            frontier = next_frontier

        for node, hop in visited.items():
            if hop == 0:
                continue
            hop_fade = fade * (1.0 - hop / (max_hops + 1))
            if hop_fade <= 0:
                continue
            c = tuple(int(c * hop_fade) for c in color)
            cv2.circle(frame, (px[node], py[node]), max(4, int(7 * hop_fade)), c, -1, cv2.LINE_AA)
            for nb in neighbors.get(node, []):
                if visited.get(nb, 99) < hop:
                    cv2.line(frame, (px[node], py[node]), (px[nb], py[nb]), c, 2, cv2.LINE_AA)

    return frame


def render_frame_radiale(t, score, width=960, height=540, orientation="verticale",
                          num_lanes=10, palette="Multicolore (per fonte)", band_colors=None,
                          position_mode="pan"):
    """Matrice 10: nessuna corsia lineare — tre ANELLI CONCENTRICI (bassi dentro,
    medi al centro, alti fuori), ognuno pilotato IN CONTINUO dall'energia reale
    della propria banda (non solo dai colpi discreti): raggio e irregolarità
    pulsano sempre, anche fra un evento e l'altro. Gli eventi aggiungono solo
    brevi spuntoni luminosi sull'anello della loro banda, nella direzione della
    loro posizione (pan/frequenza) — mai un oggetto proprio, solo un guizzo
    sopra ciò che già pulsa da sé."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    duration = max(score["duration"], 1e-6)
    cx, cy = width / 2.0, height / 2.0
    base_r = min(width, height) * 0.5

    mosaic = score["band_mosaic"]
    mode, block_seconds = mosaic["mode"], mosaic["block_seconds"]
    bc = band_colors or DEFAULT_BAND_COLORS
    seed = score["seed"]

    rings = [("bass", 0.20, 8), ("mid", 0.34, 14), ("treble", 0.46, 22)]
    n_lanes = max(3, int(num_lanes))
    n_points = max(60, n_lanes * 6)

    ring_r_by_band = {}
    for band_name, rel_r, spikes in rings:
        val = _band_continuous_value(mosaic[band_name], mode, block_seconds, t)
        ring_r = base_r * rel_r * (0.45 + 1.3 * val)  # escursione molto più marcata
        ring_r_by_band[band_name] = ring_r
        base_color = bc.get(band_name, (140, 140, 150))
        shade = 0.35 + 0.65 * val
        color = tuple(int(ch * shade) for ch in base_color)

        if orientation == "orizzontale":
            angles = np.linspace(0, np.pi, n_points)  # semicerchio: solo metà anello
            closed = False
        else:
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)  # cerchio completo
            closed = True

        # pulsazione agganciata al battito reale: un ciclo completo ogni battito,
        # invece di una velocità costante scollegata dal tempo del brano
        beat_interval = score.get("beat_interval") or 0.5
        wob = 1.0 + 0.20 * val * np.sin(angles * spikes + (t / beat_interval) * 2 * np.pi + seed * 0.01)
        xs = (cx + ring_r * wob * np.cos(angles)).astype(np.int32)
        ys = (cy + ring_r * wob * np.sin(angles)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=closed, color=color, thickness=3, lineType=cv2.LINE_AA)

    active = [e for e in score["events"] if e["t"] <= t <= e["t"] + max(e["dur"], 0.05)]
    for e in active:
        band_name = e.get("band", "mid")
        if band_name not in ring_r_by_band:
            band_name = "mid"
        prog = (t - e["t"]) / max(e["dur"], 0.05)
        fade = 1.0 - prog
        color = get_event_color(e, palette, band_colors)
        c = tuple(int(ch * fade) for ch in color)

        if orientation == "orizzontale":
            ang = np.pi * min(0.999, max(0.0, _lane_fraction(e, position_mode)))
        else:
            ang = 2 * np.pi * min(0.999, max(0.0, _lane_fraction(e, position_mode)))

        ring_r = ring_r_by_band[band_name]
        spike_r = ring_r * (1.0 + 1.1 * e["vel"] * fade)  # guizzo più lungo e visibile
        x0, y0 = int(cx + ring_r * np.cos(ang)), int(cy + ring_r * np.sin(ang))
        x1, y1 = int(cx + spike_r * np.cos(ang)), int(cy + spike_r * np.sin(ang))
        cv2.line(frame, (x0, y0), (x1, y1), c, max(3, int(3 + 4 * fade)), cv2.LINE_AA)

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


def apply_audio_degradation(stereo, sr, seed, usura_level):
    """Wow&flutter (velocità di lettura instabile), crackle, fruscio di fondo e
    micro-cadute di segnale — crescono con l'usura, come un nastro consumato."""
    if usura_level <= 0.0:
        return stereo
    N = stereo.shape[1]
    rng = np.random.RandomState(int(seed) % (2**31))

    # wow & flutter: la velocità di lettura oscilla lentamente (tipico di nastro/vinile)
    flutter_depth = 0.006 * usura_level
    lfo_freq = 0.7
    t_ax = np.arange(N) / sr
    speed_mod = 1.0 + flutter_depth * np.sin(2 * np.pi * lfo_freq * t_ax)
    warped_idx = np.clip(np.cumsum(speed_mod) - 1, 0, N - 1).astype(int)
    out_l = stereo[0][warped_idx].copy()
    out_r = stereo[1][warped_idx].copy()

    # crackle: click sparsi (polvere/graffi)
    n_clicks = int(N * 0.00004 * usura_level * 50)
    for _ in range(n_clicks):
        pos = rng.randint(0, N)
        amp = rng.uniform(0.3, 0.9) * (1 if rng.random() < 0.5 else -1)
        out_l[pos] = np.clip(out_l[pos] + amp, -1, 1)
        out_r[pos] = np.clip(out_r[pos] + amp, -1, 1)

    # fruscio di fondo costante
    hiss_amp = 0.018 * usura_level
    hiss = rng.normal(0, hiss_amp, N)
    out_l = out_l + hiss
    out_r = out_r + hiss

    # micro-cadute di segnale (dropout)
    seg_len = max(1, int(0.01 * sr))
    dropout_prob = 0.001 * usura_level
    for start in range(0, N, seg_len):
        if rng.random() < dropout_prob:
            end = min(N, start + seg_len)
            out_l[start:end] *= 0.1
            out_r[start:end] *= 0.1

    return np.clip(np.stack([out_l, out_r]), -1.0, 1.0)


def synthesize_audio_jeck(score, sr=SR, usura_level=0.0):
    """Riusa la sintesi 'pulita' di Ikeda come base, poi la consuma progressivamente."""
    base = synthesize_audio_ikeda(score, sr=sr)
    return apply_audio_degradation(base, sr, score["seed"], usura_level)


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


def synthesize_audio_molnar(score, sr=SR):
    """Un 'orologio' ritmico fisso e prevedibile (la regola che si ripete identica),
    con una nota fuori posto esattamente quando arriva una deviazione — stesso istante
    del guizzo visivo, stesso principio: la regola resta uguale finché non si rompe apposta."""
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    out_l = np.zeros(N)
    out_r = np.zeros(N)

    seed = score["seed"]
    rng = np.random.RandomState(seed + 505)
    tick_interval = 0.5  # il "battito" regolare della griglia
    scale = [0, 2, 4, 7, 9]  # pentatonica maggiore, deterministica
    pattern = [scale[rng.randint(0, len(scale))] for _ in range(8)]  # sequenza fissa che si ripete

    def note_freq(semitone_offset, base=220.0):
        return base * (2 ** (semitone_offset / 12))

    # il clock regolare — tick brevi e discreti, sempre uguali, mai reattivi all'istante
    tick_t = 0.0
    step = 0
    while tick_t < duration:
        start = int(tick_t * sr)
        dur_n = int(0.12 * sr)
        end = min(N, start + dur_n)
        if end > start:
            seg_len = end - start
            freq = note_freq(pattern[step % len(pattern)])
            seg_t = np.arange(seg_len) / sr
            env_local = np.exp(-np.linspace(0, 6, seg_len))
            wave = np.sin(2 * np.pi * freq * seg_t) * env_local * 0.18
            out_l[start:end] += wave
            out_r[start:end] += wave
        tick_t += tick_interval
        step += 1

    # le deviazioni: una nota dissonante e più forte, fuori dalla scala regolare,
    # esattamente nell'istante del guizzo visivo — sincronia totale tra vista e udito
    for e in score["deviation_events"]:
        start = int(e["t"] * sr)
        dur_n = int(0.35 * sr)
        end = min(N, start + dur_n)
        if start >= N or end <= start:
            continue
        seg_len = end - start
        dissonant_semitone = 1 if rng.random() < 0.5 else 6  # nota volutamente "sbagliata"
        freq = note_freq(dissonant_semitone, base=220.0)
        seg_t = np.arange(seg_len) / sr
        env_local = np.hanning(seg_len) if seg_len > 1 else np.ones(seg_len)
        wave = np.sin(2 * np.pi * freq * seg_t) * env_local * 0.5 * e.get("vel", 0.8)

        pan = e.get("pan", 0.0) or 0.0
        gain_l = float(np.sqrt((1.0 - pan) / 2.0))
        gain_r = float(np.sqrt((1.0 + pan) / 2.0))
        out_l[start:end] += wave * gain_l
        out_r[start:end] += wave * gain_r

    stereo = np.stack([np.clip(out_l, -1.0, 1.0), np.clip(out_r, -1.0, 1.0)])
    return stereo


def synthesize_audio_stocastico(score, sr=SR):
    """Matrice 05: ogni evento è un GLISSANDO (rampa di frequenza continua) invece
    di un tono fisso — stesso principio visivo del fascio che diverge: il suono
    non sta mai fermo su un'altezza, scivola da un punto all'altro. Il drone di
    fondo usa una deriva casuale lenta (random walk) invece della texture fissa,
    coerente con l'idea di campo stocastico continuo."""
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    out_l = np.zeros(N)
    out_r = np.zeros(N)
    t_ax = np.linspace(0, duration, N)

    env_full = np.interp(t_ax, np.linspace(0, duration, len(score["macro_envelope"])), score["macro_envelope"])

    # drone di fondo: random walk lento in frequenza (deriva stocastica continua)
    rng_drone = np.random.RandomState(score["seed"] + 707)
    steps = rng_drone.normal(0, 0.6, N // 200 + 2)
    walk = np.cumsum(steps)
    walk = np.interp(np.arange(N), np.linspace(0, N, len(walk)), walk)
    base_freq = 60.0 * (2 ** (walk / 24))  # deriva contenuta entro circa un'ottava
    phase_drone = 2 * np.pi * np.cumsum(base_freq) / sr
    drone = np.sin(phase_drone) * (0.05 + 0.08 * env_full)
    out_l += drone
    out_r += drone

    for e in score["events"]:
        start = int(e["t"] * sr)
        dur_n = max(int(0.05 * sr), int(e["dur"] * sr))
        end = min(N, start + dur_n)
        if start >= N or end <= start:
            continue
        seg_len = end - start

        seed_e = int((e["t"] * 1000) % (2 ** 31))
        rng_e = np.random.RandomState(seed_e)

        freq_start = 440.0 * (2 ** ((e["pitch"] - 69) / 12))
        # il glissando diverge di un intervallo casuale (fino a un'ottava), come le
        # linee del fascio visivo che si aprono nel tempo
        semitone_shift = rng_e.uniform(-12, 12) * (0.3 + 0.7 * e["vel"])
        freq_end = freq_start * (2 ** (semitone_shift / 12))

        seg_t = np.arange(seg_len) / sr
        inst_freq = np.linspace(freq_start, freq_end, seg_len)
        phase = 2 * np.pi * np.cumsum(inst_freq) / sr
        env_local = np.hanning(seg_len) if seg_len > 1 else np.ones(seg_len)
        signal = np.sin(phase) * env_local * e["vel"] * 0.42

        pan = e.get("pan", 0.0) or 0.0
        gain_l = float(np.sqrt((1.0 - pan) / 2.0))
        gain_r = float(np.sqrt((1.0 + pan) / 2.0))
        out_l[start:end] += signal * gain_l
        out_r[start:end] += signal * gain_r

    stereo = np.stack([np.clip(out_l, -1.0, 1.0), np.clip(out_r, -1.0, 1.0)])
    return stereo


def synthesize_audio_automaton(score, sr=SR):
    """Matrice 06: nessuna tonalità fissa scandita a evento — una NUVOLA GRANULARE
    stocastica la cui densità segue la densità viva dell'automa cellulare in
    quell'istante (esattamente come il tessuto visivo: cresce e si dirada da sé,
    non solo in reazione a un colpo). Gli eventi reali si sovrappongono come
    impulsi brevi misti tono/rumore, per restare comunque sincronizzati con la
    fioritura visiva."""
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    out_l = np.zeros(N)
    out_r = np.zeros(N)

    field = score["automaton_field"]
    steps, cols = field.shape
    density_per_step = field.mean(axis=1)
    rng = np.random.RandomState(score["seed"] + 909)

    grain_dur = 0.02
    grain_n = max(2, int(grain_dur * sr))
    win = np.hanning(grain_n)
    t_ax = np.arange(grain_n) / sr
    step_dur = duration / steps

    for i in range(steps):
        dens = float(density_per_step[i])
        n_grains = int(1 + dens * 14)
        t0 = i * step_dur
        for _ in range(n_grains):
            gt = t0 + rng.uniform(0, step_dur)
            start = int(gt * sr)
            end = min(N, start + grain_n)
            if end <= start:
                continue
            seg_len = end - start
            freq = rng.uniform(180, 3200) * (0.5 + dens)
            wave = np.sin(2 * np.pi * freq * t_ax[:seg_len]) * win[:seg_len] * (0.05 + 0.12 * dens)
            pan = rng.uniform(-0.8, 0.8)
            gain_l = float(np.sqrt((1.0 - pan) / 2.0))
            gain_r = float(np.sqrt((1.0 + pan) / 2.0))
            out_l[start:end] += wave * gain_l
            out_r[start:end] += wave * gain_r

    # eventi reali: impulsi brevi misti tono/rumore, per la sincronia audio/video
    for e in score["events"]:
        start = int(e["t"] * sr)
        dur_n = max(int(0.04 * sr), int(e["dur"] * sr * 0.6))
        end = min(N, start + dur_n)
        if start >= N or end <= start:
            continue
        seg_len = end - start
        freq = 220.0 * (2 ** ((e["pitch"] - 69) / 12))
        seg_t = np.arange(seg_len) / sr
        noise = rng.normal(0, 1, seg_len)
        tone = np.sin(2 * np.pi * freq * seg_t)
        env_local = np.hanning(seg_len) if seg_len > 1 else np.ones(seg_len)
        signal = (0.6 * tone + 0.4 * noise) * env_local * e["vel"] * 0.35

        pan = e.get("pan", 0.0) or 0.0
        gain_l = float(np.sqrt((1.0 - pan) / 2.0))
        gain_r = float(np.sqrt((1.0 + pan) / 2.0))
        out_l[start:end] += signal * gain_l
        out_r[start:end] += signal * gain_r

    stereo = np.stack([np.clip(out_l, -1.0, 1.0), np.clip(out_r, -1.0, 1.0)])
    return stereo


def synthesize_audio_datastream(score, sr=SR):
    """Matrice 07: timbro digitale a onda quadra (mai seno puro) — un clock di dati
    che scandisce tick brevissimi e intermittenti, densità legata all'inviluppo
    macro, quantizzati su una scala fissa (il 'readout' meccanico). Gli eventi
    reali si sovrappongono come tick più marcati, con l'intonazione reale
    dell'evento, per restare sincronizzati col guizzo visivo."""
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    out_l = np.zeros(N)
    out_r = np.zeros(N)
    res = len(score["macro_envelope"])
    rng = np.random.RandomState(score["seed"] + 1111)

    tick_dur = 0.02
    tick_n = max(2, int(tick_dur * sr))
    win = np.hanning(tick_n)
    tick_t = np.arange(tick_n) / sr
    scale = [0, 2, 3, 5, 7, 8, 10]  # scala minore, deliberatamente "meccanica"
    step = 0.06
    tpos = 0.0
    while tpos < duration:
        env_idx = min(res - 1, int((tpos / duration) * res))
        macro_v = float(score["macro_envelope"][env_idx])
        if rng.random() < (0.3 + 0.6 * macro_v):  # readout intermittente, non un clock pieno
            start = int(tpos * sr)
            end = min(N, start + tick_n)
            if end > start:
                seg_len = end - start
                semitone = scale[rng.randint(0, len(scale))]
                freq = 330.0 * (2 ** (semitone / 12)) * rng.choice([0.5, 1.0, 2.0])
                square = np.sign(np.sin(2 * np.pi * freq * tick_t[:seg_len]))
                sig = square * win[:seg_len] * (0.06 + 0.10 * macro_v)
                pan = rng.uniform(-0.6, 0.6)
                gain_l = float(np.sqrt((1.0 - pan) / 2.0))
                gain_r = float(np.sqrt((1.0 + pan) / 2.0))
                out_l[start:end] += sig * gain_l
                out_r[start:end] += sig * gain_r
        tpos += step

    for e in score["events"]:
        start = int(e["t"] * sr)
        dur_n = max(int(0.03 * sr), int(e["dur"] * sr * 0.5))
        end = min(N, start + dur_n)
        if start >= N or end <= start:
            continue
        seg_len = end - start
        freq = 440.0 * (2 ** ((e["pitch"] - 69) / 12))
        seg_t = np.arange(seg_len) / sr
        square = np.sign(np.sin(2 * np.pi * freq * seg_t))
        env_local = np.hanning(seg_len) if seg_len > 1 else np.ones(seg_len)
        signal = square * env_local * e["vel"] * 0.35

        pan = e.get("pan", 0.0) or 0.0
        gain_l = float(np.sqrt((1.0 - pan) / 2.0))
        gain_r = float(np.sqrt((1.0 + pan) / 2.0))
        out_l[start:end] += signal * gain_l
        out_r[start:end] += signal * gain_r

    stereo = np.stack([np.clip(out_l, -1.0, 1.0), np.clip(out_r, -1.0, 1.0)])
    return stereo


def synthesize_audio_oscilloscopio(score, sr=SR):
    """Matrice 08: un drone continuo a due oscillatori leggermente disaccordati
    (stesso rapporto armonico della figura di Lissajous visiva) che genera
    battimenti lenti — nessun ritmo scandito a evento, proprio come la traccia
    visiva non ha oggetti discreti. Gli eventi reali si sovrappongono come toni
    brevi, per restare comunque sincronizzati col guizzo del colore sulla curva."""
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    t_ax = np.arange(N) / sr

    seed = score["seed"]
    freq_x = 3 + (seed % 5)
    freq_y = 2 + ((seed // 5) % 5)
    base_f = 55.0

    env_full = np.interp(t_ax, np.linspace(0, duration, len(score["macro_envelope"])), score["macro_envelope"])
    osc_x = np.sin(2 * np.pi * base_f * freq_x * 0.5 * t_ax)
    osc_y = np.sin(2 * np.pi * base_f * freq_y * 0.5 * t_ax * 1.003)  # leggero detuning: battimenti
    drone = (osc_x * 0.5 + osc_y * 0.5) * (0.07 + 0.10 * env_full)
    out_l = drone.copy()
    out_r = drone * 0.9 + (osc_y - osc_x) * 0.03

    for e in score["events"]:
        start = int(e["t"] * sr)
        dur_n = max(int(0.05 * sr), int(e["dur"] * sr))
        end = min(N, start + dur_n)
        if start >= N or end <= start:
            continue
        seg_len = end - start
        freq = 220.0 * (2 ** ((e["pitch"] - 69) / 12))
        seg_t = np.arange(seg_len) / sr
        tone = np.sin(2 * np.pi * freq * seg_t)
        env_local = np.hanning(seg_len) if seg_len > 1 else np.ones(seg_len)
        signal = tone * env_local * e["vel"] * 0.30

        pan = e.get("pan", 0.0) or 0.0
        gain_l = float(np.sqrt((1.0 - pan) / 2.0))
        gain_r = float(np.sqrt((1.0 + pan) / 2.0))
        out_l[start:end] += signal * gain_l
        out_r[start:end] += signal * gain_r

    stereo = np.stack([np.clip(out_l, -1.0, 1.0), np.clip(out_r, -1.0, 1.0)])
    return stereo


def synthesize_audio_rete(score, sr=SR):
    """Matrice 09: nessun drone di fondo, nessuna texture continua — il circuito è
    silenzioso finché non viene toccato. Ogni evento è un PLUCK a decadimento
    esponenziale (come una corda pizzicata), con una seconda armonica leggera per
    un timbro metallico da circuito. La risonanza dura più a lungo dell'evento
    stesso, coerente con l'impulso che si propaga sulla rete visiva."""
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    out_l = np.zeros(N)
    out_r = np.zeros(N)

    for e in score["events"]:
        start = int(e["t"] * sr)
        dur_n = max(int(0.25 * sr), int(e["dur"] * sr * 1.5))
        end = min(N, start + dur_n)
        if start >= N or end <= start:
            continue
        seg_len = end - start
        freq = 220.0 * (2 ** ((e["pitch"] - 69) / 12))
        seg_t = np.arange(seg_len) / sr
        decay = np.exp(-seg_t * (3.0 + 4.0 * (1.0 - e["vel"])))
        tone = np.sin(2 * np.pi * freq * seg_t) * decay
        tone += 0.3 * np.sin(2 * np.pi * freq * 2.01 * seg_t) * decay
        signal = tone * e["vel"] * 0.4

        pan = e.get("pan", 0.0) or 0.0
        gain_l = float(np.sqrt((1.0 - pan) / 2.0))
        gain_r = float(np.sqrt((1.0 + pan) / 2.0))
        out_l[start:end] += signal * gain_l
        out_r[start:end] += signal * gain_r

    stereo = np.stack([np.clip(out_l, -1.0, 1.0), np.clip(out_r, -1.0, 1.0)])
    return stereo


def synthesize_audio_radiale(score, sr=SR):
    """Matrice 10: tre oscillatori simultanei (grave/medio/acuto), ognuno modulato
    IN CONTINUO dall'energia reale della propria banda — esattamente come i tre
    anelli visivi, il drone risponde sempre a bassi/medi/alti, non solo ai colpi
    discreti. Gli eventi reali si sovrappongono come toni brevi per la sincronia
    col guizzo visivo sull'anello corrispondente."""
    duration = max(score["duration"], 0.1)
    N = int(duration * sr)
    t_ax = np.arange(N) / sr

    mosaic = score["band_mosaic"]
    mode, block_seconds = mosaic["mode"], mosaic["block_seconds"]

    step = 0.05
    n_steps = int(duration / step) + 2
    times_steps = np.arange(n_steps) * step
    bass_vals = np.array([_band_continuous_value(mosaic["bass"], mode, block_seconds, tt) for tt in times_steps])
    mid_vals = np.array([_band_continuous_value(mosaic["mid"], mode, block_seconds, tt) for tt in times_steps])
    treble_vals = np.array([_band_continuous_value(mosaic["treble"], mode, block_seconds, tt) for tt in times_steps])
    bass_env = np.interp(t_ax, times_steps, bass_vals)
    mid_env = np.interp(t_ax, times_steps, mid_vals)
    treble_env = np.interp(t_ax, times_steps, treble_vals)

    osc_bass = np.sin(2 * np.pi * 60.0 * t_ax) * (0.10 + 0.20 * bass_env)
    osc_mid = np.sin(2 * np.pi * 220.0 * t_ax) * (0.06 + 0.14 * mid_env)
    osc_treble = np.sin(2 * np.pi * 880.0 * t_ax) * (0.04 + 0.10 * treble_env)
    drone = osc_bass + osc_mid + osc_treble
    out_l = drone.copy()
    out_r = drone.copy()

    for e in score["events"]:
        start = int(e["t"] * sr)
        dur_n = max(int(0.05 * sr), int(e["dur"] * sr))
        end = min(N, start + dur_n)
        if start >= N or end <= start:
            continue
        seg_len = end - start
        freq = 220.0 * (2 ** ((e["pitch"] - 69) / 12))
        seg_t = np.arange(seg_len) / sr
        tone = np.sin(2 * np.pi * freq * seg_t)
        env_local = np.hanning(seg_len) if seg_len > 1 else np.ones(seg_len)
        signal = tone * env_local * e["vel"] * 0.3

        pan = e.get("pan", 0.0) or 0.0
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
    """Report nel formato standard Loop507 (:: MOTORE / EFFETTO / TECHNICAL LOG SHEET).
    Bilingue: italiano (come sempre) seguito dalla versione inglese."""
    meta = MODULES[module_id]
    counts = {}
    for e in score["events"]:
        counts[e["source"]] = counts.get(e["source"], 0) + 1

    band_counts = {}
    for e in score["events"]:
        if "band" in e:
            band_counts[e["band"]] = band_counts.get(e["band"], 0) + 1

    analisi = ["Automa Cellulare", "Rumore Multi-Ottava"]
    if counts.get("audio", 0) > 0:
        analisi += ["Onset Detection", "Band Split (bassi/medi/alti)", "RMS Envelope"]
    if counts.get("video", 0) > 0:
        analisi.append("Scene Cut Detection")
    if counts.get("midi", 0) > 0:
        analisi.append("MIDI Parsing")
    if module_id == "02":
        analisi.append("Macro Block Segmentation")
    if module_id == "03":
        analisi.append("Grid Deviation Detection")
    if module_id == "04":
        analisi.append("Tape Decay Tracking")
    if module_id == "05":
        analisi.append("Poisson Point Field / Divergent Line Sweep")
    if module_id == "06":
        analisi.append("2D Cellular Automaton Field / Granular Density Synthesis")
    if module_id == "07":
        analisi.append("Glyph Stream Rendering / Square-Wave Data Clock")
    if module_id == "08":
        analisi.append("Lissajous Parametric Curve / Phosphor Persistence Trail")
    if module_id == "09":
        analisi.append("Fixed Network Topology / Impulse Propagation")
    if module_id == "10":
        analisi.append("Continuous Band Envelope Following / Concentric Ring Modulation")

    vol_num = vol if vol is not None else abs(score["seed"]) % 99
    n_frames = int(round(score["duration"] * FPS))
    ts_clean = params["timestamp"].replace("/", "").replace(":", "").replace(" ", "_")

    def build(lang):
        it = (lang == "it")
        labels = {"ca": "Procedurale" if it else "Procedural", "midi": "MIDI",
                  "audio": "Audio", "video": "Video"}
        fonti_attive = [labels[k] for k in ("midi", "audio", "video") if counts.get(k, 0) > 0]
        fonti_str = " + ".join(fonti_attive) if fonti_attive else \
            ("Nessuna (generazione pura)" if it else "None (pure generation)")

        nome = meta["nome"] if it else meta["nome_en"]
        processo = meta["processo"] if it else meta["processo_en"]
        motore_tag = meta["motore_tag"] if it else meta["motore_tag_en"]
        quote = meta["quote"] if it else meta["quote_en"]

        r = []
        r.append(f"[BEATGLITCH_MATRICE_ENGINE_{module_id}] // VOL_{vol_num:02d} // H.264 // DATA_FRAGMENT")
        r.append(f":: MOTORE: matrice_engine_{module_id} [v1.0 — {motore_tag}]" if it else
                  f":: ENGINE: matrice_engine_{module_id} [v1.0 — {motore_tag}]")
        r.append(f":: EFFETTO: {meta['effetto']} — Regola {params['rule']}" if it else
                  f":: EFFECT: {meta['effetto']} — Rule {params['rule']}")
        r.append(f":: ANALISI: {' / '.join(analisi)}" if it else
                  f":: ANALYSIS: {' / '.join(analisi)}")
        r.append(f":: PROCESSO: {processo} — Fonti: {fonti_str}" if it else
                  f":: PROCESS: {processo} — Sources: {fonti_str}")
        r.append("")
        r.append(f'"{quote}"')
        r.append("")
        r.append(":: TECHNICAL LOG SHEET:")
        r.append(f"* File: matrice_output_{ts_clean}")
        r.append(f"* Modulo: {module_id} — {nome}" if it else f"* Module: {module_id} — {nome}")
        r.append(f"* Seed (utente): {params['seed']}" if it else f"* Seed (user): {params['seed']}")
        r.append(f"* Seed effettivo: {score['seed']} (perturbato dal contenuto delle fonti esterne)" if it else
                  f"* Effective seed: {score['seed']} (perturbed by external source content)")
        r.append(f"* Rendering: {n_frames} frame @ {FPS}fps" if it else
                  f"* Rendering: {n_frames} frames @ {FPS}fps")
        r.append(f"* Risoluzione: {params['resolution']}" if it else f"* Resolution: {params['resolution']}")
        r.append(f"* Durata: {score['duration']:.1f}s" if it else f"* Duration: {score['duration']:.1f}s")
        if score.get("bpm"):
            r.append(f"* BPM rilevato: {score['bpm']:.1f}" if it else f"* Detected BPM: {score['bpm']:.1f}")
        else:
            r.append("* BPM: non rilevato (ipotesi neutra 120 BPM per le cadenze)" if it else
                      "* BPM: not detected (neutral 120 BPM assumption used for cadences)")
        if module_id == "02":
            mosaic = score["band_mosaic"]
            if it:
                modo_str = "energia audio reale" if mosaic["mode"] == "audio" else "procedurale (nessun audio)"
                r.append(f"* Mosaico: 4x3 celle (bassi/medi/alti), ogni {mosaic['block_seconds']:.1f}s, {modo_str}")
            else:
                modo_str = "real audio energy" if mosaic["mode"] == "audio" else "procedural (no audio)"
                r.append(f"* Mosaic: 4x3 cells (bass/mid/treble), every {mosaic['block_seconds']:.1f}s, {modo_str}")
        if module_id == "03":
            n_dev = len(score["deviation_events"])
            n_dev_audio = sum(1 for e in score["deviation_events"] if e["source"] == "audio")
            if it:
                r.append(f"* Deviazioni: {n_dev} totali ({n_dev_audio} da colpi audio forti, "
                          f"{n_dev - n_dev_audio} procedurali)")
            else:
                r.append(f"* Deviations: {n_dev} total ({n_dev_audio} from strong audio hits, "
                          f"{n_dev - n_dev_audio} procedural)")
        if module_id == "04":
            if it:
                r.append(f"* Riproduzioni finora: {params.get('usura_count', '?')}")
                r.append(f"* Livello usura: {params.get('usura_level', 0)*100:.0f}%")
            else:
                r.append(f"* Plays so far: {params.get('usura_count', '?')}")
                r.append(f"* Wear level: {params.get('usura_level', 0)*100:.0f}%")
        r.append(f"* Colonna sonora: {params['audio_mode']}" if it else f"* Soundtrack: {params['audio_mode']}")
        r.append(f"* Eventi totali: {len(score['events'])}" if it else
                  f"* Total events: {len(score['events'])}")
        r.append(f"* Eventi Procedurali: {counts.get('ca', 0)}" if it else
                  f"* Procedural events: {counts.get('ca', 0)}")
        if counts.get("midi", 0) > 0:
            r.append(f"* Eventi MIDI: {counts.get('midi', 0)}" if it else f"* MIDI events: {counts.get('midi', 0)}")
        if counts.get("audio", 0) > 0:
            r.append(f"* Eventi Audio: {counts.get('audio', 0)}" if it else
                      f"* Audio events: {counts.get('audio', 0)}")
            if it:
                r.append(f"* Bilanciamento Frequenze: bassi {band_counts.get('bass', 0)} | "
                          f"medi {band_counts.get('mid', 0)} | alti {band_counts.get('treble', 0)}")
            else:
                r.append(f"* Frequency balance: bass {band_counts.get('bass', 0)} | "
                          f"mid {band_counts.get('mid', 0)} | treble {band_counts.get('treble', 0)}")
        if counts.get("video", 0) > 0:
            r.append(f"* Eventi Video (tagli scena): {counts.get('video', 0)}" if it else
                      f"* Video events (scene cuts): {counts.get('video', 0)}")
        r.append("")
        r.append(f":: Regia e Algoritmo: {brand}" if it else f":: Direction and Algorithm: {brand}")
        r.append("")
        r.append(f"#generativeart #proceduralart #digitalminimalism {meta['hashtag']}")
        r.append("#computationalminimalism #brutalistart #glitchart #audiovisual")
        r.append("#experimentalvideo #beatglitch")
        return r

    righe_it = build("it")
    righe_en = build("en")

    righe = list(righe_it)
    righe.append("")
    righe.append(":: ENGLISH VERSION " + "-" * 40)
    righe.append("")
    righe += righe_en

    return "\n".join(righe)






import base64
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

RENDER_FNS = {"01": render_frame_ikeda, "02": render_frame_henke, "03": render_frame_molnar,
              "04": render_frame_jeck, "05": render_frame_stocastico, "06": render_frame_automaton,
              "07": render_frame_datastream, "08": render_frame_oscilloscopio, "09": render_frame_rete,
              "10": render_frame_radiale}
SYNTH_FNS = {"01": synthesize_audio_ikeda, "02": synthesize_audio_henke, "03": synthesize_audio_molnar,
             "04": synthesize_audio_jeck, "05": synthesize_audio_stocastico, "06": synthesize_audio_automaton,
             "07": synthesize_audio_datastream, "08": synthesize_audio_oscilloscopio, "09": synthesize_audio_rete,
             "10": synthesize_audio_radiale}
render_frame_fn = RENDER_FNS[module_id]
synthesize_audio_fn = SYNTH_FNS[module_id]

if module_id == "02":
    st.caption("Modulo 02: un mosaico di 12 blocchi (4 colonne × bassi/medi/alti) "
               "si ricompone in base all'energia reale delle tre bande di frequenza.")
elif module_id == "03":
    st.caption("Modulo 03: una griglia rigida resta quasi sempre identica. Devia "
               "solo sui colpi audio davvero forti (o raramente da sé, senza audio).")
elif module_id == "04":
    _usura_count_preview = load_usura_count()
    st.caption(f"Modulo 04: questa istanza ha già generato {_usura_count_preview} volte. "
               f"Più cresce il numero, più il segnale è consumato — non si azzera da sé.")
elif module_id == "05":
    st.caption("Modulo 05: nessuna griglia e nessuna cella. Ogni evento apre un fascio "
               "di linee divergenti che attraversa l'intero fotogramma, su un fondo di "
               "pulviscolo stocastico.")
elif module_id == "06":
    st.caption("Modulo 06: un automa cellulare 2D scorre come tessuto continuo sullo "
               "sfondo, mutando anche senza input. Gli eventi sbocciano sopra come "
               "fioriture morbide di colore, non barre né linee.")
elif module_id == "07":
    st.caption("Modulo 07: un flusso continuo di caratteri scorre in colonne, come un "
               "readout che non si ferma mai. Gli eventi accendono col proprio colore "
               "i caratteri della corsia in cui accadono.")
elif module_id == "08":
    st.caption("Modulo 08: nessun oggetto discreto. Una sola curva continua (figura "
               "di Lissajous) si disegna con una scia fosforescente; gli eventi "
               "colorano solo il tratto di curva corrispondente al loro istante.")
elif module_id == "09":
    st.caption("Modulo 09: l'unica rete a topologia fissa. Nodi e collegamenti "
               "restano invisibili (nero puro) finché un evento non accende il "
               "nodo più vicino e la corrente si propaga lungo la rete.")
elif module_id == "10":
    st.caption("Modulo 10: tre anelli concentrici (bassi/medi/alti) pulsano sempre, "
               "in continuo, con l'energia reale delle tre bande — non solo ai colpi. "
               "Gli eventi aggiungono solo brevi spuntoni luminosi sopra l'anello.")







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

    if module_id == "01":
        orientamento_label = st.radio("Orientamento linee", ["Verticali", "Orizzontali", "Verticali + Orizzontali"],
                                       horizontal=True)
        ORIENTAMENTI = {"Verticali": "verticale", "Orizzontali": "orizzontale",
                         "Verticali + Orizzontali": "misto"}
        orientamento = ORIENTAMENTI[orientamento_label]
        if orientamento == "misto":
            st.caption("In modalità mista: bassi verticali (sostenuti), alti orizzontali (rapidi), "
                       "medi alternati nel tempo.")
        num_lanes = st.slider("Numero di linee", 1, 24, 10, key="num_lanes_01")
        posizione_label = st.radio("Posizione orizzontale", ["Pan stereo reale", "Frequenza (bassi←→alti)"],
                                    horizontal=True, key="posizione_01",
                                    help="Pan reale: segue la posizione stereo vera del colpo — se il mix è "
                                         "centrato/mono, le barre restano vicine al centro qualunque sia il "
                                         "numero di linee. Frequenza: ignora il pan, distribuisce sempre "
                                         "sull'intera larghezza (bassi a sinistra, alti a destra).")
        position_mode = "pan" if posizione_label == "Pan stereo reale" else "frequenza"

        usa_banda_01 = st.checkbox("Colori separati per bassi/medi/alti", value=False, key="banda_toggle_01")
        band_colors = None
        if usa_banda_01:
            palette = "Per banda (bassi/medi/alti)"
            st.caption("Un colore per ciascuna banda di frequenza (solo eventi audio; "
                       "le altre fonti restano grigio chiaro).")
            c_bassi = st.color_picker("Bassi", "#EB2828", key="ikeda_bassi")
            c_medi = st.color_picker("Medi", "#FFAA00", key="ikeda_medi")
            c_alti = st.color_picker("Alti", "#00C8FF", key="ikeda_alti")

            def _hex_to_rgb(h):
                h = h.lstrip("#")
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

            band_colors = {"bass": _hex_to_rgb(c_bassi), "mid": _hex_to_rgb(c_medi),
                            "treble": _hex_to_rgb(c_alti)}
        else:
            palette_options = [k for k in PALETTES.keys() if k != "Per banda (bassi/medi/alti)"]
            palette = st.selectbox("Palette colore", palette_options, key="palette_01")

        st.markdown("---")
        show_source_legend = st.checkbox("Mostra legenda colori fonte", value=True,
                                          disabled=(palette != "Multicolore (per fonte)"))
        grid_cols, accent_color, deviation_sensitivity, deviation_min_gap = 10, (235, 40, 60), 0.6, 1.0
    elif module_id == "02":
        # Modulo 02: il mosaico è sempre colorato per banda — niente orientamento/
        # linee/palette (non esistono più grani da colorare individualmente)
        st.caption("Il mosaico si ricompone in base all'energia reale di bassi/medi/alti "
                   "(o a sequenze procedurali se non carichi audio). Scegli i colori delle tre righe.")
        c_bassi = st.color_picker("Bassi (riga in basso)", "#EB2828")
        c_medi = st.color_picker("Medi (riga centrale)", "#FFAA00")
        c_alti = st.color_picker("Alti (riga in alto)", "#00C8FF")

        def _hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

        band_colors = {"bass": _hex_to_rgb(c_bassi), "mid": _hex_to_rgb(c_medi),
                        "treble": _hex_to_rgb(c_alti)}
        orientamento, num_lanes, palette = "verticale", 10, "Multicolore (per fonte)"
        grid_cols, accent_color, deviation_sensitivity, deviation_min_gap = 10, (235, 40, 60), 0.6, 1.0
        position_mode = "pan"
        show_source_legend = False
    elif module_id == "03":
        # Modulo 03: griglia rigida, quasi sempre uniforme — densità, colori
        # per banda e ritmo delle deviazioni sono personalizzabili
        st.caption("La griglia resta identica quasi sempre. Il colore della deviazione "
                   "dipende dalla banda del colpo che l'ha causata (bassi/medi/alti).")
        grid_cols = st.slider("Densità griglia (colonne)", 4, 20, 10)
        c_bassi = st.color_picker("Deviazione da bassi", "#EB2828")
        c_medi = st.color_picker("Deviazione da medi", "#FFAA00")
        c_alti = st.color_picker("Deviazione da alti", "#00C8FF")
        c_procedurale = st.color_picker("Deviazione procedurale (senza audio)", "#CCCCCC")
        deviation_sensitivity = st.slider(
            "Sensibilità deviazioni", 0.3, 0.9, 0.6, step=0.05,
            help="Più basso = più colpi audio fanno scattare una deviazione (più frequenti). "
                 "Più alto = solo i colpi davvero più forti (più rare)."
        )
        deviation_min_gap = st.slider(
            "Distanza minima tra deviazioni (in battiti)", 0.25, 4.0, 1.0, step=0.25,
            help="Calibrata sul BPM reale rilevato dall'audio caricato: un valore di 1.0 "
                 "significa 'non prima di un battito'. Un brano veloce (es. techno) avrà "
                 "quindi deviazioni più ravvicinate in automatico, uno lento più diradate. "
                 "Senza audio (o BPM non rilevato) si assume un'ipotesi neutra di 120 BPM."
        )

        def _hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

        band_colors = {"bass": _hex_to_rgb(c_bassi), "mid": _hex_to_rgb(c_medi),
                        "treble": _hex_to_rgb(c_alti)}
        accent_color = _hex_to_rgb(c_procedurale)
        orientamento, num_lanes, palette = "verticale", 10, "Multicolore (per fonte)"
        position_mode = "pan"
        show_source_legend = False
    elif module_id == "04":
        # Modulo 04: riusa lo stesso linguaggio visivo/sonoro del modulo 01, ma lo
        # consuma in base a quante volte è già stato generato — persistente su disco
        st.caption("Il degrado cresce col numero di generazioni fatte finora su questa "
                   "istanza (satura dopo 50). Non c'è modo di 'pulirlo' se non azzerarlo.")
        _usura_count = load_usura_count()
        _usura_level_reale = usura_level_from_count(_usura_count)
        st.metric("Riproduzioni finora", _usura_count, help="Persistente finché il container resta attivo.")
        st.progress(_usura_level_reale, text=f"Usura: {_usura_level_reale*100:.0f}%")

        preview_override = st.checkbox("Anteprima: forza un livello di usura diverso", value=False)
        if preview_override:
            usura_level_effettivo = st.slider("Livello usura (anteprima)", 0.0, 1.0, _usura_level_reale, step=0.05)
        else:
            usura_level_effettivo = _usura_level_reale

        if st.button("🔄 Azzera usura (irreversibile)"):
            save_usura_count(0)
            st.rerun()

        st.markdown("---")
        num_lanes = st.slider("Numero di linee", 1, 24, 10, key="num_lanes_04")
        orientamento_label_04 = st.radio("Orientamento linee", ["Verticali", "Orizzontali", "Verticali + Orizzontali"],
                                          horizontal=True, key="orientamento_04")
        ORIENTAMENTI_04 = {"Verticali": "verticale", "Orizzontali": "orizzontale",
                            "Verticali + Orizzontali": "misto"}
        orientamento = ORIENTAMENTI_04[orientamento_label_04]

        usa_banda_04 = st.checkbox("Colori separati per bassi/medi/alti", value=False, key="banda_toggle_04")
        band_colors = None
        if usa_banda_04:
            palette = "Per banda (bassi/medi/alti)"
            c_bassi_j = st.color_picker("Bassi", "#EB2828", key="jeck_bassi")
            c_medi_j = st.color_picker("Medi", "#FFAA00", key="jeck_medi")
            c_alti_j = st.color_picker("Alti", "#00C8FF", key="jeck_alti")

            def _hex_to_rgb(h):
                h = h.lstrip("#")
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

            band_colors = {"bass": _hex_to_rgb(c_bassi_j), "mid": _hex_to_rgb(c_medi_j),
                            "treble": _hex_to_rgb(c_alti_j)}
        else:
            palette_options_j = [k for k in PALETTES.keys() if k != "Per banda (bassi/medi/alti)"]
            palette = st.selectbox("Palette colore", palette_options_j, key="palette_04")

        grid_cols, accent_color = 10, (235, 40, 60)
        position_mode = "pan"
        deviation_sensitivity, deviation_min_gap = 0.6, 1.0
        show_source_legend = (palette == "Multicolore (per fonte)")
    elif module_id == "05":
        # Modulo 05: campo stocastico continuo — niente griglia, niente celle.
        # Riusa lo stesso sistema di colori/comandi degli altri moduli, ma la resa
        # visiva è fasci di linee divergenti su un fondo di pulviscolo.
        st.caption("Ogni evento apre un fascio di linee che diverge nel tempo, invece "
                   "di una barra confinata in una corsia. 'Numero di linee' qui indica "
                   "quante linee compongono ciascun fascio.")
        orientamento_label_05 = st.radio("Orientamento fasci", ["Verticali", "Orizzontali", "Verticali + Orizzontali"],
                                          horizontal=True, key="orientamento_05")
        ORIENTAMENTI_05 = {"Verticali": "verticale", "Orizzontali": "orizzontale",
                            "Verticali + Orizzontali": "misto"}
        orientamento = ORIENTAMENTI_05[orientamento_label_05]
        num_lanes = st.slider("Numero di linee per fascio", 1, 24, 10, key="num_lanes_05")
        posizione_label_05 = st.radio("Posizione orizzontale", ["Pan stereo reale", "Frequenza (bassi←→alti)"],
                                       horizontal=True, key="posizione_05")
        position_mode = "pan" if posizione_label_05 == "Pan stereo reale" else "frequenza"

        usa_banda_05 = st.checkbox("Colori separati per bassi/medi/alti", value=False, key="banda_toggle_05")
        band_colors = None
        if usa_banda_05:
            palette = "Per banda (bassi/medi/alti)"
            c_bassi_5 = st.color_picker("Bassi", "#EB2828", key="stoc_bassi")
            c_medi_5 = st.color_picker("Medi", "#FFAA00", key="stoc_medi")
            c_alti_5 = st.color_picker("Alti", "#00C8FF", key="stoc_alti")

            def _hex_to_rgb(h):
                h = h.lstrip("#")
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

            band_colors = {"bass": _hex_to_rgb(c_bassi_5), "mid": _hex_to_rgb(c_medi_5),
                            "treble": _hex_to_rgb(c_alti_5)}
        else:
            palette_options_5 = [k for k in PALETTES.keys() if k != "Per banda (bassi/medi/alti)"]
            palette = st.selectbox("Palette colore", palette_options_5, key="palette_05")

        grid_cols, accent_color, deviation_sensitivity, deviation_min_gap = 10, (235, 40, 60), 0.6, 1.0
        show_source_legend = (palette == "Multicolore (per fonte)")
    elif module_id == "06":
        # Modulo 06: il tessuto di fondo (automa cellulare 2D) muta da sé nel tempo,
        # indipendentemente dagli eventi — gli eventi aggiungono solo le fioriture
        # di colore sopra la texture continua.
        st.caption("Il tessuto di fondo cresce e si dirada da solo, anche in assenza "
                   "di eventi. 'Numero di linee' qui regola la scala delle fioriture "
                   "di colore rispetto al tessuto.")
        orientamento_label_06 = st.radio("Orientamento fioriture", ["Verticali", "Orizzontali", "Verticali + Orizzontali"],
                                          horizontal=True, key="orientamento_06")
        ORIENTAMENTI_06 = {"Verticali": "verticale", "Orizzontali": "orizzontale",
                            "Verticali + Orizzontali": "misto"}
        orientamento = ORIENTAMENTI_06[orientamento_label_06]
        num_lanes = st.slider("Numero di linee (scala fioriture)", 1, 24, 10, key="num_lanes_06")
        posizione_label_06 = st.radio("Posizione orizzontale", ["Pan stereo reale", "Frequenza (bassi←→alti)"],
                                       horizontal=True, key="posizione_06")
        position_mode = "pan" if posizione_label_06 == "Pan stereo reale" else "frequenza"

        usa_banda_06 = st.checkbox("Colori separati per bassi/medi/alti", value=False, key="banda_toggle_06")
        band_colors = None
        if usa_banda_06:
            palette = "Per banda (bassi/medi/alti)"
            c_bassi_6 = st.color_picker("Bassi", "#EB2828", key="auto_bassi")
            c_medi_6 = st.color_picker("Medi", "#FFAA00", key="auto_medi")
            c_alti_6 = st.color_picker("Alti", "#00C8FF", key="auto_alti")

            def _hex_to_rgb(h):
                h = h.lstrip("#")
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

            band_colors = {"bass": _hex_to_rgb(c_bassi_6), "mid": _hex_to_rgb(c_medi_6),
                            "treble": _hex_to_rgb(c_alti_6)}
        else:
            palette_options_6 = [k for k in PALETTES.keys() if k != "Per banda (bassi/medi/alti)"]
            palette = st.selectbox("Palette colore", palette_options_6, key="palette_06")

        grid_cols, accent_color, deviation_sensitivity, deviation_min_gap = 10, (235, 40, 60), 0.6, 1.0
        show_source_legend = (palette == "Multicolore (per fonte)")
    elif module_id == "07":
        # Modulo 07: il flusso di caratteri scorre da sé in ogni corsia — gli eventi
        # accendono solo il colore della corsia in cui accadono, non disegnano forme.
        st.caption("Il flusso di caratteri scorre sempre, anche senza eventi (in grigio "
                   "spento). 'Numero di linee' qui è il numero di corsie del flusso.")
        orientamento_label_07 = st.radio("Direzione flusso", ["Verticale", "Orizzontale", "Verticale + Orizzontale"],
                                          horizontal=True, key="orientamento_07")
        ORIENTAMENTI_07 = {"Verticale": "verticale", "Orizzontale": "orizzontale",
                            "Verticale + Orizzontale": "misto"}
        orientamento = ORIENTAMENTI_07[orientamento_label_07]
        num_lanes = st.slider("Numero di corsie", 1, 200, 60, key="num_lanes_07")
        posizione_label_07 = st.radio("Posizione orizzontale", ["Pan stereo reale", "Frequenza (bassi←→alti)"],
                                       horizontal=True, key="posizione_07")
        position_mode = "pan" if posizione_label_07 == "Pan stereo reale" else "frequenza"

        usa_banda_07 = st.checkbox("Colori separati per bassi/medi/alti", value=False, key="banda_toggle_07")
        band_colors = None
        if usa_banda_07:
            palette = "Per banda (bassi/medi/alti)"
            c_bassi_7 = st.color_picker("Bassi", "#EB2828", key="ds_bassi")
            c_medi_7 = st.color_picker("Medi", "#FFAA00", key="ds_medi")
            c_alti_7 = st.color_picker("Alti", "#00C8FF", key="ds_alti")

            def _hex_to_rgb(h):
                h = h.lstrip("#")
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

            band_colors = {"bass": _hex_to_rgb(c_bassi_7), "mid": _hex_to_rgb(c_medi_7),
                            "treble": _hex_to_rgb(c_alti_7)}
        else:
            palette_options_7 = [k for k in PALETTES.keys() if k != "Per banda (bassi/medi/alti)"]
            palette = st.selectbox("Palette colore", palette_options_7, key="palette_07")

        grid_cols, accent_color, deviation_sensitivity, deviation_min_gap = 10, (235, 40, 60), 0.6, 1.0
        show_source_legend = (palette == "Multicolore (per fonte)")
    elif module_id == "08":
        # Modulo 08: nessuna corsia audio-reattiva nel senso classico — la curva è
        # unica e continua. 'Numero di linee' qui regola solo la lunghezza della
        # scia fosforescente (persistenza), non un numero di corsie.
        st.caption("Non ci sono corsie: la curva è una sola. 'Numero di linee' qui "
                   "allunga o accorcia la scia fosforescente che la segue.")
        orientamento_label_08 = st.radio("Proporzione figura", ["Verticale", "Orizzontale", "Bilanciata"],
                                          horizontal=True, key="orientamento_08")
        ORIENTAMENTI_08 = {"Verticale": "verticale", "Orizzontale": "orizzontale", "Bilanciata": "misto"}
        orientamento = ORIENTAMENTI_08[orientamento_label_08]
        num_lanes = st.slider("Lunghezza scia (persistenza)", 1, 24, 10, key="num_lanes_08")

        usa_banda_08 = st.checkbox("Colori separati per bassi/medi/alti", value=False, key="banda_toggle_08")
        band_colors = None
        if usa_banda_08:
            palette = "Per banda (bassi/medi/alti)"
            c_bassi_8 = st.color_picker("Bassi", "#EB2828", key="osc_bassi")
            c_medi_8 = st.color_picker("Medi", "#FFAA00", key="osc_medi")
            c_alti_8 = st.color_picker("Alti", "#00C8FF", key="osc_alti")

            def _hex_to_rgb(h):
                h = h.lstrip("#")
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

            band_colors = {"bass": _hex_to_rgb(c_bassi_8), "mid": _hex_to_rgb(c_medi_8),
                            "treble": _hex_to_rgb(c_alti_8)}
        else:
            palette_options_8 = [k for k in PALETTES.keys() if k != "Per banda (bassi/medi/alti)"]
            palette = st.selectbox("Palette colore", palette_options_8, key="palette_08")

        grid_cols, accent_color, deviation_sensitivity, deviation_min_gap = 10, (235, 40, 60), 0.6, 1.0
        show_source_legend = (palette == "Multicolore (per fonte)")
    elif module_id == "09":
        # Modulo 09: unica matrice a topologia fissa — nodi e archi non cambiano
        # mai posizione. 'Numero di linee' qui è il numero di nodi della rete
        # (moltiplicato internamente per una densità di collegamenti fissa).
        st.caption("La rete resta invisibile (nero puro) finché un evento non "
                   "accende il nodo più vicino e la corrente si propaga lungo i "
                   "collegamenti vicini.")
        orientamento_label_09 = st.radio("Assi corsia/banda", ["Standard", "Assi scambiati"],
                                          horizontal=True, key="orientamento_09")
        orientamento = "verticale" if orientamento_label_09 == "Standard" else "orizzontale"
        num_lanes = st.slider("Numero di nodi", 3, 100, 20, key="num_lanes_09")
        posizione_label_09 = st.radio("Posizione sulla rete", ["Pan stereo reale", "Frequenza (bassi←→alti)"],
                                       horizontal=True, key="posizione_09")
        position_mode = "pan" if posizione_label_09 == "Pan stereo reale" else "frequenza"

        usa_banda_09 = st.checkbox("Colori separati per bassi/medi/alti", value=False, key="banda_toggle_09")
        band_colors = None
        if usa_banda_09:
            palette = "Per banda (bassi/medi/alti)"
            c_bassi_9 = st.color_picker("Bassi", "#EB2828", key="rete_bassi")
            c_medi_9 = st.color_picker("Medi", "#FFAA00", key="rete_medi")
            c_alti_9 = st.color_picker("Alti", "#00C8FF", key="rete_alti")

            def _hex_to_rgb(h):
                h = h.lstrip("#")
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

            band_colors = {"bass": _hex_to_rgb(c_bassi_9), "mid": _hex_to_rgb(c_medi_9),
                            "treble": _hex_to_rgb(c_alti_9)}
        else:
            palette_options_9 = [k for k in PALETTES.keys() if k != "Per banda (bassi/medi/alti)"]
            palette = st.selectbox("Palette colore", palette_options_9, key="palette_09")

        grid_cols, accent_color, deviation_sensitivity, deviation_min_gap = 10, (235, 40, 60), 0.6, 1.0
        show_source_legend = (palette == "Multicolore (per fonte)")
    else:
        # Modulo 10: tre anelli concentrici sempre pulsanti con l'energia reale di
        # bassi/medi/alti. 'Numero di linee' qui è la risoluzione angolare degli
        # anelli (quanti segmenti li compongono).
        st.caption("Gli anelli pulsano sempre con le tre bande, anche senza eventi — "
                   "più forte è l'energia della banda, più l'anello si allarga. "
                   "'Numero di linee' qui regola la risoluzione angolare degli anelli.")
        orientamento_label_10 = st.radio("Forma anelli", ["Cerchio completo", "Semicerchio"],
                                          horizontal=True, key="orientamento_10")
        orientamento = "verticale" if orientamento_label_10 == "Cerchio completo" else "orizzontale"
        num_lanes = st.slider("Risoluzione anelli", 20, 120, 40, key="num_lanes_10")
        posizione_label_10 = st.radio("Posizione guizzi", ["Pan stereo reale", "Frequenza (bassi←→alti)"],
                                       horizontal=True, key="posizione_10")
        position_mode = "pan" if posizione_label_10 == "Pan stereo reale" else "frequenza"

        usa_banda_10 = st.checkbox("Colori separati per bassi/medi/alti", value=True, key="banda_toggle_10")
        band_colors = None
        if usa_banda_10:
            palette = "Per banda (bassi/medi/alti)"
            c_bassi_10 = st.color_picker("Bassi", "#EB2828", key="rad_bassi")
            c_medi_10 = st.color_picker("Medi", "#FFAA00", key="rad_medi")
            c_alti_10 = st.color_picker("Alti", "#00C8FF", key="rad_alti")

            def _hex_to_rgb(h):
                h = h.lstrip("#")
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

            band_colors = {"bass": _hex_to_rgb(c_bassi_10), "mid": _hex_to_rgb(c_medi_10),
                            "treble": _hex_to_rgb(c_alti_10)}
        else:
            palette_options_10 = [k for k in PALETTES.keys() if k != "Per banda (bassi/medi/alti)"]
            palette = st.selectbox("Palette colore", palette_options_10, key="palette_10")

        grid_cols, accent_color, deviation_sensitivity, deviation_min_gap = 10, (235, 40, 60), 0.6, 1.0
        show_source_legend = (palette == "Multicolore (per fonte)")

# ------------------------------------------------------------
# ESTRAZIONE — ogni input presente alimenta un ruolo diverso
# ------------------------------------------------------------
midi_events, audio_events, audio_env, video_events, video_env = None, None, None, None, None
audio_raw = None
audio_band_envelopes = None
detected_bpm = None
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
        audio_events, audio_env, audio_dur, audio_raw, audio_band_envelopes, detected_bpm = extract_from_audio(audio_path)
        durations.append(audio_dur)
        if detected_bpm:
            st.sidebar.success(f"Audio: {len(audio_events)} onset rilevati — BPM stimato: {detected_bpm:.1f}")
        else:
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
        # blocco macro (Henke/Radiale): una battuta intera (4/4) se conosciamo il BPM
        # reale, invece di una costante fissa in secondi — i blocchi cambiano sui
        # tempi forti del brano, non a un ritmo arbitrario
        macro_block_seconds_effective = (60.0 / detected_bpm) * 4 if detected_bpm else 3.0
        score = build_score(
            duration=duration, seed=int(seed), rule=rule,
            midi_events=midi_events,
            audio_events=audio_events, audio_env=audio_env,
            video_events=video_events, video_env=video_env,
            audio_band_envelopes=audio_band_envelopes,
            macro_block_seconds=macro_block_seconds_effective,
            deviation_strong_threshold=deviation_sensitivity,
            deviation_min_gap=deviation_min_gap,
            bpm=detected_bpm,
        )
        st.write(f"Matrice pronta — {len(score['events'])} eventi totali.")
        if score["bpm"]:
            st.write(f"BPM rilevato: {score['bpm']:.1f} — cadenze delle matrici calibrate su questo tempo.")
        else:
            st.write("Nessun BPM rilevato (audio assente o non conclusivo) — "
                      "ipotesi neutra di 120 BPM usata per le cadenze.")

        st.write("Costruzione colonna sonora finale...")
        N = int(score["duration"] * SR)
        synth_kwargs = {"usura_level": usura_level_effettivo} if module_id == "04" else {}
        generated = synthesize_audio_fn(score, sr=SR, **synth_kwargs)  # shape (2, N)

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

        extra_kwargs = {}
        if module_id == "03":
            extra_kwargs = {"grid_cols": grid_cols, "accent_color": accent_color}
        elif module_id == "01":
            extra_kwargs = {"position_mode": position_mode}
        elif module_id == "04":
            extra_kwargs = {"usura_level": usura_level_effettivo}
        elif module_id == "05":
            extra_kwargs = {"position_mode": position_mode}
        elif module_id == "06":
            extra_kwargs = {"position_mode": position_mode}
        elif module_id == "07":
            extra_kwargs = {"position_mode": position_mode}
        elif module_id == "09":
            extra_kwargs = {"position_mode": position_mode}
        elif module_id == "10":
            extra_kwargs = {"position_mode": position_mode}

        def make_frame(t):
            return render_frame_fn(t, score, width=export_w, height=export_h,
                                    orientation=orientamento, num_lanes=num_lanes,
                                    palette=palette, band_colors=band_colors, **extra_kwargs)

        clip = VideoClip(make_frame, duration=score["duration"])
        clip = clip.with_fps(FPS) if hasattr(clip, "with_fps") else clip.set_fps(FPS)
        audio_clip = AudioFileClip(t_wav.name)
        clip = clip.with_audio(audio_clip) if hasattr(clip, "with_audio") else clip.set_audio(audio_clip)

        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        clip.write_videofile(
            out_path, codec="libx264", audio_codec="aac", fps=FPS, logger=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-color_range", "pc",
                           "-colorspace", "bt709", "-color_primaries", "bt709",
                           "-color_trc", "bt709"]
        )

        # l'usura avanza solo con generazioni reali, non con l'anteprima forzata
        if module_id == "04" and not preview_override:
            save_usura_count(_usura_count + 1)

        st.write("Generazione report...")
        ts_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{MODULE_FILENAME_BASE}_{ts_compact}"
        report_params = {
            "seed": int(seed), "rule": rule, "duration": score["duration"],
            "resolution": f"{export_w}x{export_h}", "audio_mode": audio_mode,
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "usura_count": (_usura_count + 1) if module_id == "04" and not preview_override else
                           (_usura_count if module_id == "04" else None),
            "usura_level": usura_level_effettivo if module_id == "04" else None,
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

    # st.video serve il file tramite l'endpoint media di Streamlit invece di
    # incorporarlo come base64 nel messaggio: con usura/distruzione alta il file
    # è molto più pesante (rumore ad alta entropia = compressione H.264 inefficace)
    # e l'embed base64 poteva superare il limite di dimensione messaggio, dando
    # errore. st.video non ha questo limite.
    col_v1, col_v2, col_v3 = st.columns([1, 2, 1])
    with col_v2:
        with open(res["video_path"], "rb") as f:
            st.video(f.read())

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
