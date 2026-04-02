import streamlit as st
import os
import numpy as np
import tempfile
import traceback
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize
import io

# ─────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────

def detect_cuts(video_path: str, min_cut_interval: float = 0.3) -> list[float]:
    """Rileva i tagli nel video analizzando le differenze tra frame."""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        fps = min(clip.fps, 10)  # max 10fps per leggerezza
        times = np.arange(0, duration, 1.0 / fps)
        cuts = [0.0]
        prev_frame = None
        for t in times:
            try:
                frame = clip.get_frame(t).mean(axis=(0, 1))
                if prev_frame is not None:
                    diff = np.abs(frame - prev_frame).mean()
                    if diff > 25 and (t - cuts[-1]) >= min_cut_interval:
                        cuts.append(round(t, 3))
                prev_frame = frame
            except Exception:
                continue
        clip.close()
        return cuts
    except Exception as e:
        st.error(f"Errore rilevamento tagli: {e}")
        return [0.0]


def load_audio_from_video(video_path: str, sr: int = 44100):
    """Estrae l'audio dal video come array numpy."""
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None, sr
        audio_array = clip.audio.to_soundarray(fps=sr)
        clip.close()
        if audio_array.ndim == 2:
            return audio_array.T, sr  # (channels, samples)
        return audio_array[np.newaxis, :], sr
    except Exception:
        return None, sr


def load_audio_file(audio_path: str, sr: int = 44100):
    """Carica un file audio esterno."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        return y, sr
    except Exception as e:
        st.error(f"Errore caricamento audio: {e}")
        return None, sr


# ─────────────────────────────────────────────
#  EFFETTI AUDIO
# ─────────────────────────────────────────────

def apply_stutter(segment: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Ripete rapidamente un pezzo del segmento."""
    stutter_len = int(sr * 0.05 * (1 + intensity))
    stutter_len = min(stutter_len, segment.shape[-1])
    piece = segment[..., :stutter_len]
    repeats = int(2 + intensity * 4)
    stuttered = np.concatenate([piece] * repeats, axis=-1)
    return stuttered[..., :segment.shape[-1]]


def apply_reverse(segment: np.ndarray) -> np.ndarray:
    """Inverte il segmento."""
    return segment[..., ::-1].copy()


def apply_loop(segment: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Fa loopare una piccola parte del segmento."""
    loop_len = int(sr * 0.1 * (1 + intensity))
    loop_len = min(loop_len, segment.shape[-1])
    piece = segment[..., :loop_len]
    total = segment.shape[-1]
    loops = np.tile(piece, int(np.ceil(total / loop_len)))
    return loops[..., :total]


def apply_jump(segment: np.ndarray, sr: int) -> np.ndarray:
    """Salta a un punto casuale del segmento."""
    n = segment.shape[-1]
    if n < 2:
        return segment
    jump_point = np.random.randint(0, n // 2)
    return np.concatenate([segment[..., jump_point:], segment[..., :jump_point]], axis=-1)


def apply_glitch_effect(segment: np.ndarray, sr: int, effect: str, intensity: float) -> np.ndarray:
    """Applica l'effetto scelto al segmento."""
    if effect == "stutter":
        return apply_stutter(segment, sr, intensity)
    elif effect == "reverse":
        return apply_reverse(segment)
    elif effect == "loop":
        return apply_loop(segment, sr, intensity)
    elif effect == "jump":
        return apply_jump(segment, sr)
    return segment


def generate_synthetic_audio(cuts: list[float], duration: float, sr: int = 44100,
                               synth_type: str = "beat", intensity: float = 0.5) -> np.ndarray:
    """Genera audio sintetico basato sui tagli del video."""
    total_samples = int(duration * sr)
    audio = np.zeros((2, total_samples))

    for cut in cuts:
        start = int(cut * sr)
        if start >= total_samples:
            continue

        if synth_type == "beat":
            # Kick sintetico
            t = np.linspace(0, 0.1, int(sr * 0.1))
            kick = np.sin(2 * np.pi * 60 * t) * np.exp(-30 * t)
            kick *= 0.8 * (0.5 + intensity)
            end = min(start + len(kick), total_samples)
            audio[:, start:end] += kick[:end - start]

        elif synth_type == "noise":
            # Burst di rumore
            burst_len = int(sr * 0.05 * (1 + intensity))
            noise = np.random.randn(burst_len) * 0.3 * intensity
            end = min(start + burst_len, total_samples)
            audio[:, start:end] += noise[:end - start]

        elif synth_type == "tone":
            # Tono breve
            freq = 220 * (1 + intensity)
            tone_len = int(sr * 0.08)
            t = np.linspace(0, 0.08, tone_len)
            tone = np.sin(2 * np.pi * freq * t) * np.exp(-20 * t) * 0.6
            end = min(start + tone_len, total_samples)
            audio[:, start:end] += tone[:end - start]

    # Normalizza
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    return audio


def remix_audio_on_cuts(audio: np.ndarray, cuts: list[float], duration: float,
                         sr: int, effects: list[str], intensity: float) -> np.ndarray:
    """Taglia e remixa l'audio sui punti di taglio del video."""
    total_samples = int(duration * sr)
    result = np.zeros((2, total_samples))

    cut_times = cuts + [duration]
    for i in range(len(cut_times) - 1):
        start_t = cut_times[i]
        end_t = cut_times[i + 1]
        start_s = int(start_t * sr)
        end_s = int(end_t * sr)

        # Estrai segmento dall'audio sorgente (con wrap se più corto)
        audio_len = audio.shape[-1]
        seg_len = end_s - start_s
        src_start = start_s % audio_len
        src_end = src_start + seg_len

        if src_end <= audio_len:
            segment = audio[:, src_start:src_end].copy()
        else:
            part1 = audio[:, src_start:]
            part2 = audio[:, :(src_end - audio_len) % audio_len]
            segment = np.concatenate([part1, part2], axis=-1)

        # Applica effetto casuale tra quelli scelti
        if effects and np.random.random() < (0.3 + intensity * 0.5):
            effect = np.random.choice(effects)
            segment = apply_glitch_effect(segment, sr, effect, intensity)

        # Assicura lunghezza corretta
        if segment.shape[-1] > seg_len:
            segment = segment[:, :seg_len]
        elif segment.shape[-1] < seg_len:
            pad = np.zeros((2, seg_len - segment.shape[-1]))
            segment = np.concatenate([segment, pad], axis=-1)

        end_s = min(end_s, total_samples)
        result[:, start_s:end_s] = segment[:, :end_s - start_s]

    peak = np.abs(result).max()
    if peak > 0:
        result = result / peak * 0.9
    return result


def mix_audio(audio1: np.ndarray, audio2: np.ndarray, ratio: float) -> np.ndarray:
    """Mixa due array audio con un rapporto dato (0=solo audio1, 1=solo audio2)."""
    min_len = min(audio1.shape[-1], audio2.shape[-1])
    mixed = audio1[:, :min_len] * (1 - ratio) + audio2[:, :min_len] * ratio
    peak = np.abs(mixed).max()
    if peak > 0:
        mixed = mixed / peak * 0.9
    return mixed


# ─────────────────────────────────────────────
#  EXPORT
# ─────────────────────────────────────────────

def export_audio(audio: np.ndarray, sr: int, output_path_wav: str, output_path_mp3: str):
    """Esporta l'audio in WAV lossless e MP3 320kbps."""
    # WAV
    sf.write(output_path_wav, audio.T, sr, subtype='PCM_24')

    # MP3 320kbps via pydub
    seg = AudioSegment.from_wav(output_path_wav)
    seg = normalize(seg)
    seg.export(output_path_mp3, format="mp3", bitrate="320k")


def export_video_guide(video_path: str, audio: np.ndarray, sr: int, output_path: str):
    """Esporta il video guida a bassa risoluzione con il nuovo audio."""
    try:
        clip = VideoFileClip(video_path)
        # Ridimensiona a 360p
        w, h = clip.size
        scale = 360 / h
        new_w = int(w * scale) // 2 * 2
        clip_resized = clip.resize((new_w, 360))

        # Crea clip audio
        audio_T = audio.T.astype(np.float32)
        duration = len(audio_T) / sr
        audio_clip = AudioArrayClip(audio_T, fps=sr)
        audio_clip = audio_clip.subclip(0, min(duration, clip_resized.duration))

        final = clip_resized.set_audio(audio_clip)
        final.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp_beatglitch_audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None,
            fps=24
        )
        clip.close()
        final.close()
        return True
    except Exception as e:
        st.error(f"Errore export video guida: {e}\n{traceback.format_exc()}")
        return False


# ─────────────────────────────────────────────
#  UI PRINCIPALE
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="BeatGlitch by Loop507",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎵 BeatGlitch")
    st.markdown("##### by Loop507")
    st.markdown("Analizza i tagli del tuo video remix e genera un audio sincronizzato con effetti glitch, stutter e beat sintetici.")
    st.markdown("---")

    # ── CARICAMENTO FILE ──
    st.subheader("📂 Carica i file")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_video = st.file_uploader(
            "🎬 Video remix (obbligatorio)",
            type=["mp4", "mov", "avi", "mkv"],
            help="Il video da cui verranno rilevati i tagli."
        )
    with col2:
        uploaded_audio = st.file_uploader(
            "🎵 Audio/Musica (opzionale)",
            type=["mp3", "wav", "aac", "ogg", "flac"],
            help="La tua musica da remixare. Se non carichi nulla, verrà generato audio sintetico."
        )

    if not uploaded_video:
        st.info("Carica almeno il video remix per iniziare.")
        return

    # Salva file temporanei
    tmp = tempfile.gettempdir()
    video_path = os.path.join(tmp, f"bg_video_{uploaded_video.name}")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    audio_path = None
    if uploaded_audio:
        audio_path = os.path.join(tmp, f"bg_audio_{uploaded_audio.name}")
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.read())

    # Preview video
    st.video(video_path, start_time=0)

    st.markdown("---")
    st.subheader("⚙️ Parametri")

    # ── MODALITÀ ──
    mode = st.radio(
        "Modalità audio:",
        ["🎛️ Generativa", "🎚️ Remix Audio", "🔀 Ibrida"],
        help="""
        - **Generativa**: crea audio sintetico dai tagli del video
        - **Remix Audio**: remixa la tua musica sui tagli (richiede audio caricato)
        - **Ibrida**: mixa audio sintetico + tua musica (richiede audio caricato)
        """
    )

    if mode != "🎛️ Generativa" and not uploaded_audio:
        st.warning("⚠️ Per le modalità Remix e Ibrida devi caricare un file audio!")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Effetti Glitch**")
        use_stutter = st.checkbox("Stutter", value=True)
        use_reverse = st.checkbox("Reverse", value=True)
        use_loop = st.checkbox("Loop", value=False)
        use_jump = st.checkbox("Jump", value=False)

    with col2:
        intensity = st.slider(
            "Intensità effetti",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="0 = effetti leggeri, 1 = effetti estremi"
        )
        if mode in ["🎛️ Generativa", "🔀 Ibrida"]:
            synth_type = st.selectbox(
                "Tipo suono sintetico",
                ["beat", "noise", "tone"],
                help="beat=kick drum, noise=rumore, tone=tono sinusoidale"
            )
        if mode == "🔀 Ibrida":
            mix_ratio = st.slider(
                "Mix Sintetico / Tua Musica",
                min_value=0.0, max_value=1.0, value=0.4, step=0.05,
                help="0 = solo sintetico, 1 = solo tua musica"
            )

    min_cut_interval = st.slider(
        "Intervallo minimo tra tagli (secondi)",
        min_value=0.1, max_value=2.0, value=0.3, step=0.1,
        help="Evita di rilevare tagli troppo ravvicinati"
    )

    st.markdown("---")

    # ── AVVIO ──
    if st.button("🚀 Genera BeatGlitch", use_container_width=True):
        effects = []
        if use_stutter: effects.append("stutter")
        if use_reverse: effects.append("reverse")
        if use_loop: effects.append("loop")
        if use_jump: effects.append("jump")

        if not effects:
            st.warning("Seleziona almeno un effetto glitch!")
            return

        progress = st.progress(0)
        status = st.empty()
        SR = 44100

        try:
            # 1. Rileva tagli
            status.text("🔍 Rilevamento tagli nel video...")
            cuts = detect_cuts(video_path, min_cut_interval)
            progress.progress(20)
            st.info(f"✂️ Rilevati **{len(cuts)}** tagli nel video.")

            # 2. Durata video
            with VideoFileClip(video_path) as vc:
                duration = vc.duration

            # 3. Carica/genera audio
            status.text("🎵 Preparazione audio...")
            synth_audio = None
            remix_audio = None

            if mode in ["🎛️ Generativa", "🔀 Ibrida"]:
                synth_audio = generate_synthetic_audio(cuts, duration, SR, synth_type, intensity)

            if mode in ["🎚️ Remix Audio", "🔀 Ibrida"] and audio_path:
                raw_audio, _ = load_audio_file(audio_path, SR)
                if raw_audio is not None:
                    remix_audio = remix_audio_on_cuts(raw_audio, cuts, duration, SR, effects, intensity)

            progress.progress(60)

            # 4. Mix finale
            status.text("🔀 Mix finale...")
            if mode == "🎛️ Generativa":
                final_audio = synth_audio
            elif mode == "🎚️ Remix Audio":
                final_audio = remix_audio
            else:  # Ibrida
                final_audio = mix_audio(synth_audio, remix_audio, mix_ratio)

            progress.progress(75)

            # 5. Export audio
            status.text("💾 Esportazione audio...")
            out_wav = os.path.join(tmp, "beatglitch_output.wav")
            out_mp3 = os.path.join(tmp, "beatglitch_output.mp3")
            export_audio(final_audio, SR, out_wav, out_mp3)
            progress.progress(85)

            # 6. Export video guida
            status.text("🎬 Creazione video guida 360p...")
            out_video = os.path.join(tmp, "beatglitch_guide.mp4")
            video_ok = export_video_guide(video_path, final_audio, SR, out_video)
            progress.progress(100)
            status.empty()

            # ── RISULTATI ──
            st.success("🎉 **BeatGlitch completato con successo!**")
            st.markdown("---")
            st.subheader("📥 Download")

            col1, col2, col3 = st.columns(3)

            with col1:
                with open(out_wav, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica WAV (lossless)",
                        f.read(),
                        file_name="beatglitch_audio.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )

            with col2:
                with open(out_mp3, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica MP3 320kbps",
                        f.read(),
                        file_name="beatglitch_audio.mp3",
                        mime="audio/mpeg",
                        use_container_width=True
                    )

            with col3:
                if video_ok and os.path.exists(out_video):
                    with open(out_video, "rb") as f:
                        st.download_button(
                            "⬇️ Scarica Video Guida 360p",
                            f.read(),
                            file_name="beatglitch_guide.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )

            st.markdown("---")
            st.subheader("📊 Statistiche")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tagli rilevati", len(cuts))
            with col2:
                st.metric("Durata audio", f"{duration:.1f}s")
            with col3:
                st.metric("Sample rate", f"{SR}Hz")

            # Timeline tagli
            with st.expander("📋 Timeline tagli"):
                timeline = "Tagli rilevati:\n"
                for i, cut in enumerate(cuts):
                    timeline += f"{i+1:03d} - {cut:.3f}s\n"
                st.code(timeline, language="text")
                st.download_button(
                    "📄 Scarica Timeline",
                    timeline,
                    file_name="beatglitch_timeline.txt",
                    mime="text/plain"
                )

            if video_ok and os.path.exists(out_video):
                st.markdown("---")
                st.subheader("👁️ Anteprima Video Guida")
                st.video(out_video)

        except Exception as e:
            st.error(f"❌ Errore: {str(e)}")
            with st.expander("🔍 Dettagli errore"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
