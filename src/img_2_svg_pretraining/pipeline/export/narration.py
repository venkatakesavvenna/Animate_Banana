"""Narration audio: text-to-speech, and muxing it against the frames.

Two jobs, kept separate because they fail for different reasons and only the
first one costs money:

1. `synthesize_narration` turns the narrative writer's JSONL into one WAV per
   timestamp, via Gemini TTS.
2. `mux_narrated_video` pairs those clips with the exported frames and writes
   a single MP4 where each frame is held for exactly as long as its clip runs.

The pairing is by timestamp index, and the timestamps come from the sequence
traversal -- one playback step, one frame, one clip. A step whose narrative is
null still occupies its index; it simply gets a short silence rather than being
dropped, because dropping it would shift every later frame and desync the rest
of the video.

Two environment details drive the implementation:

- **ffmpeg comes from `imageio-ffmpeg`, not the system.** The container has no
  ffmpeg on PATH; the bundled binary is what the existing MP4 export already
  runs through. `_ffmpeg()` resolves it, preferring a system install if one
  ever appears.
- **There is no `ffprobe`.** The reference implementation shells out to it for
  clip durations. That binary is not bundled, so durations are read straight
  from the WAV header instead -- which is exact for PCM, and one less
  subprocess per clip.

API keys come from the same pool the chat backends use (`backends/keys.py`),
so TTS rotates across the same rate-limited free-tier keys rather than
carrying a hardcoded one.
"""
from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

# Matches the reference implementation's output; Gemini TTS returns raw PCM at
# this rate and these values are what make the WAV header describe it truly.
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1

DEFAULT_TTS_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_VOICE = "Callirrhoe"
DEFAULT_STYLE_DIRECTION = "[fast-paced, energetic, professional narrator voice]"

# Held on a step with no narration. Long enough to read the frame, short
# enough not to feel like a stall.
SILENT_STEP_SECONDS = 1.5


class NarrationError(Exception):
    pass


@dataclass
class Clip:
    timestamp: int
    path: Path
    duration: float
    text: str | None


def _ffmpeg() -> str:
    """The ffmpeg binary to use.

    Prefers a system install, falls back to the one `imageio-ffmpeg` bundles --
    which in this container is the only one that exists.
    """
    from shutil import which

    found = which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise NarrationError(
            "no ffmpeg available: install it, or `pip install imageio-ffmpeg`"
        ) from e


def write_wav(path: Path, pcm: bytes) -> Path:
    """Wrap raw PCM in a WAV header so players and ffmpeg can read it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    return path


def write_silence(path: Path, seconds: float = SILENT_STEP_SECONDS) -> Path:
    """A silent clip, so a step with no narration still holds its frame."""
    return write_wav(path, b"\x00" * int(SAMPLE_RATE * SAMPLE_WIDTH * seconds))


def wav_duration(path: Path) -> float:
    """Clip length in seconds, read from the WAV header.

    Exact for PCM, and avoids `ffprobe`, which is not bundled with
    `imageio-ffmpeg` and is absent from this container.
    """
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        return handle.getnframes() / float(rate) if rate else 0.0


def _pcm_from_interaction(interaction) -> bytes:
    """Pull raw PCM bytes out of a TTS response, decoding base64 if needed."""
    import base64

    audio = getattr(getattr(interaction, "output_audio", None), "data", None)
    if audio is None:
        raise NarrationError("TTS response carried no audio")
    return base64.b64decode(audio) if isinstance(audio, str) else audio


def synthesize_narration(
    jsonl_path: Path,
    out_dir: Path,
    api_keys: list[str],
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_VOICE,
    style_direction: str = DEFAULT_STYLE_DIRECTION,
    force: bool = False,
) -> list[Clip]:
    """Generate one WAV per narration timestamp. Returns the clips in order.

    Existing clips are reused unless `force`, so a partially-completed run --
    the normal outcome when a free-tier key hits its daily cap mid-sample --
    resumes instead of re-paying for what already succeeded.
    """
    from google import genai

    from ..backends.keys import KeyRing

    jsonl_path, out_dir = Path(jsonl_path), Path(out_dir)
    if not jsonl_path.exists():
        raise NarrationError(
            f"no narration script at {jsonl_path}. Run the narrate stage first.")
    if not api_keys:
        raise NarrationError(
            "no API key for TTS. Set GOOGLE_API_KEY or add a Google row to api_keys.csv.")

    out_dir.mkdir(parents=True, exist_ok=True)
    ring = KeyRing(api_keys)
    clients: dict[str, object] = {}

    def client_for(key: str):
        if key not in clients:
            clients[key] = genai.Client(api_key=key)
        return clients[key]

    entries = [json.loads(line) for line in
               jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    clips: list[Clip] = []
    for entry in entries:
        timestamp = entry.get("timestamp")
        text = entry.get("text")
        path = out_dir / f"audio_ts_{timestamp}.wav"

        if path.exists() and not force:
            clips.append(Clip(timestamp, path, wav_duration(path), text))
            continue

        if not text or not str(text).strip():
            write_silence(path)
            clips.append(Clip(timestamp, path, wav_duration(path), None))
            continue

        pcm = _generate_one(ring, client_for, model, voice,
                            f"{style_direction}\n{text}")
        write_wav(path, pcm)
        clips.append(Clip(timestamp, path, wav_duration(path), text))

    return clips


def _generate_one(ring, client_for, model: str, voice: str, prompt: str,
                  attempts: int = 6) -> bytes:
    """One TTS call, rotating keys past rate limits.

    Mirrors the chat backends' behaviour: a 429 marks that key exhausted and
    moves to the next rather than burning the retry budget on a dead key.
    """
    last_error: Exception | None = None
    for _ in range(attempts):
        key = ring.current
        try:
            interaction = client_for(key).interactions.create(
                model=model,
                input=prompt,
                response_format={"type": "audio"},
                generation_config={"speech_config": [{"voice": voice}]},
            )
            return _pcm_from_interaction(interaction)
        except Exception as e:  # SDK raises provider-specific error types
            last_error = e
            message = str(e).lower()
            if "429" in message or "quota" in message or "resource_exhausted" in message:
                ring.mark_exhausted()
                ring.rotate()
                continue
            raise NarrationError(f"TTS call failed: {e}") from e
    raise NarrationError(f"TTS rate-limited on every key: {last_error}")


def align_frames_to_clips(frames: list[Path], clips: list[Clip]) -> list[tuple[Path, float]]:
    """Assign each frame a hold time so the video tracks the narration.

    Frames and narration steps are NOT 1:1 in practice. The designer emits
    sub-frames within a step for smooth transitions, so a 4-step sequence can
    render as 17 pages. Observed across real samples: 7 steps/41 frames,
    4 steps/17, and also exact matches -- the ratio is per-sample and not
    knowable in advance.

    So frames are distributed across steps proportionally and each step's
    clip duration is split evenly among the frames that belong to it. The
    narration therefore stays anchored to the step it describes, whatever the
    frame budget, and the total video length always equals the total audio
    length.

    Fewer frames than steps is possible too (a step that renders no new page);
    those steps get no frame of their own and their audio rides on the
    previous frame, which is the only option that keeps audio and video the
    same length.
    """
    if not frames or not clips:
        raise NarrationError("need both frames and clips to align")

    total_frames, total_steps = len(frames), len(clips)
    timeline: list[tuple[Path, float]] = []
    carried = 0.0   # duration of steps that got no frame of their own

    for index, clip in enumerate(clips):
        start = index * total_frames // total_steps
        stop = (index + 1) * total_frames // total_steps
        span = frames[start:stop]
        if not span:
            # This step contributes no frame. Its audio still has to play, so
            # hand the time to a neighbouring frame -- the previous one when
            # there is one, otherwise the next frame to be emitted. Dropping
            # it would leave the audio longer than the video and desync
            # everything downstream of this step.
            if timeline:
                path, held = timeline[-1]
                timeline[-1] = (path, held + clip.duration)
            else:
                carried += clip.duration
            continue
        share = (clip.duration + carried) / len(span)
        carried = 0.0
        timeline.extend((frame, share) for frame in span)

    return timeline


def mux_narrated_video(frames: list[Path], clips: list[Clip], out_path: Path,
                       fps: int = 25, max_width: int | None = 1920) -> Path:
    """Render the frames held in time with the narration, audio muxed in."""
    frames, out_path = [Path(f) for f in frames], Path(out_path)
    if not frames:
        raise NarrationError("no frames to mux")
    if not clips:
        raise NarrationError("no narration clips to mux")

    timeline = align_frames_to_clips(frames, clips)
    ffmpeg = _ffmpeg()
    work = out_path.parent / f"_narration_{out_path.stem}"
    work.mkdir(parents=True, exist_ok=True)

    try:
        concat = work / "frames.txt"
        with open(concat, "w", encoding="utf-8") as handle:
            for frame, hold in timeline:
                handle.write(f"file '{frame.resolve()}'\n")
                handle.write(f"duration {hold:.6f}\n")
            # The concat demuxer ignores the duration on the final entry, so
            # the last frame must be repeated for its hold time to apply.
            handle.write(f"file '{timeline[-1][0].resolve()}'\n")

        # Force even dimensions (libx264 requires them); optionally downscale
        # first, since 300-dpi TikZ frames can otherwise exceed the level a
        # phone's hardware decoder accepts.
        scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        if max_width:
            scale = (f"scale='if(gt(iw,{max_width}),{max_width},iw)':-2,"
                     f"scale=trunc(iw/2)*2:trunc(ih/2)*2")

        silent = work / "silent.mp4"
        _run([
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
            # Constant frame rate: sparse VFR output produces near-keyframeless
            # streams whose duration index makes players freeze or desync.
            "-fps_mode", "cfr", "-r", str(fps),
            "-vf", scale, "-pix_fmt", "yuv420p",
            "-g", str(fps),               # a keyframe a second, for seeking
            "-profile:v", "high", "-level", "4.0",
            str(silent),
        ])

        audio_list = work / "audio.txt"
        audio_list.write_text(
            "".join(f"file '{c.path.resolve()}'\n" for c in clips), encoding="utf-8")
        full_audio = work / "narration.wav"
        # Re-encode to a common PCM format; concatenating WAVs that differ
        # even slightly in format glitches otherwise.
        _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(audio_list),
              "-c", "pcm_s16le", str(full_audio)])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _run([
            ffmpeg, "-y", "-i", str(silent), "-i", str(full_audio),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            # Move the index atom to the front, without which many in-app and
            # mobile players will not start the file at all.
            "-movflags", "+faststart", "-shortest", str(out_path),
        ])
        return out_path
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise NarrationError(
            f"{Path(cmd[0]).name} failed:\n{(proc.stderr or proc.stdout)[-2000:]}")
