"""Audio metadata and signal statistics via ffprobe/ffmpeg (CPU only).

Nothing here keeps audio: each function decodes into memory, reduces to a few
numbers and discards the samples.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

CLIP_LEVEL = 0.999
FRAME_SECONDS = 0.02
# A frame is "silent" when it is this far below the file's loud (p95) frames.
SILENCE_BELOW_P95_DB = 40.0
SPECTRUM_FRAME = 2048


class AudioError(RuntimeError):
    """ffprobe/ffmpeg could not read a file."""


def probe(path: Path) -> dict:
    """Container and first-audio-stream metadata."""

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise AudioError(f"ffprobe failed ({result.returncode})")
    payload = json.loads(result.stdout)
    streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise AudioError("no audio stream")
    stream, fmt = streams[0], payload.get("format", {})
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": stream.get("channels"),
        "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
        "duration_seconds": float(fmt["duration"]) if fmt.get("duration") else None,
        "audio_streams": len(streams),
    }


def decode(path: Path, sample_rate: int, channels: int, start: float | None = None,
           seconds: float | None = None) -> np.ndarray:
    """Decode to float32, shape (samples, channels)."""

    command = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(path)]
    if seconds is not None:
        command += ["-t", f"{seconds:.3f}"]
    command += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels), "-ar",
                str(sample_rate), "-"]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg decode failed ({result.returncode})")
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    return samples.reshape(-1, channels)


def _db(value: float) -> float | None:
    return round(20 * np.log10(value), 2) if value > 0 else None


def level_stats(mono: np.ndarray, sample_rate: int) -> dict:
    """Peak, RMS, clipping, silence share and frame dynamic range of a mono signal."""

    if mono.size == 0:
        return {"decoded_seconds": 0.0}
    frame = max(1, int(sample_rate * FRAME_SECONDS))
    usable = mono[: mono.size - mono.size % frame]
    frame_rms = np.sqrt(np.mean(usable.reshape(-1, frame).astype(np.float64) ** 2, axis=1))
    frame_db = 20 * np.log10(np.maximum(frame_rms, 1e-10))
    p95 = float(np.percentile(frame_db, 95)) if frame_db.size else float("nan")
    p10 = float(np.percentile(frame_db, 10)) if frame_db.size else float("nan")
    return {
        "decoded_seconds": round(mono.size / sample_rate, 3),
        "peak_dbfs": _db(float(np.max(np.abs(mono)))),
        "rms_dbfs": _db(float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))),
        "clipped_sample_ratio": round(float(np.mean(np.abs(mono) >= CLIP_LEVEL)), 6),
        "silence_ratio": round(float(np.mean(frame_db < p95 - SILENCE_BELOW_P95_DB)), 4),
        "frame_dynamic_range_db": round(p95 - p10, 2),
    }


def rolloff_hz(mono: np.ndarray, sample_rate: int, share: float = 0.99) -> float | None:
    """Frequency below which `share` of the mean power spectrum of loud frames lies.

    MP3 encoders low-pass the signal, so this exposes the effective bandwidth.
    """

    n = SPECTRUM_FRAME
    count = mono.size // n
    if count == 0:
        return None
    frames = mono[: count * n].reshape(count, n).astype(np.float64)
    energy = np.mean(frames**2, axis=1)
    loud = frames[energy >= np.percentile(energy, 50)] if count > 1 else frames
    spectrum = np.mean(np.abs(np.fft.rfft(loud * np.hanning(n), axis=1)) ** 2, axis=0)
    cumulative = np.cumsum(spectrum)
    if cumulative[-1] <= 0:
        return None
    index = int(np.searchsorted(cumulative, share * cumulative[-1]))
    return round(index * sample_rate / n, 1)


def stereo_stats(stereo: np.ndarray) -> dict:
    """Inter-channel correlation and level gap; ~1.0 / ~0 dB means dual mono."""

    left, right = stereo[:, 0].astype(np.float64), stereo[:, 1].astype(np.float64)
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return {"channel_correlation": None, "channel_level_gap_db": None}
    correlation = float(np.corrcoef(left, right)[0, 1])
    gap = 20 * np.log10(max(left.std(), 1e-12) / max(right.std(), 1e-12))
    return {
        "channel_correlation": round(correlation, 4),
        "channel_level_gap_db": round(float(gap), 2),
    }
