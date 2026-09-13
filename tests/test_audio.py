import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lit_sp_en import audio

SR = 16000


def _tone(freq: float, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_level_stats_detects_silence_and_clipping() -> None:
    signal = np.concatenate([_tone(440, 1.0), np.zeros(SR, dtype=np.float32)])
    signal[:10] = 1.0
    stats = audio.level_stats(signal, SR)
    assert stats["decoded_seconds"] == 2.0
    assert 0.45 <= stats["silence_ratio"] <= 0.55
    assert stats["clipped_sample_ratio"] > 0
    assert stats["peak_dbfs"] == 0.0


def test_rolloff_tracks_band_limit() -> None:
    assert audio.rolloff_hz(_tone(1000, 2.0), SR) == pytest.approx(1000, abs=20)


def test_stereo_stats_dual_mono_vs_independent() -> None:
    tone = _tone(300, 1.0)
    dual = np.stack([tone, tone], axis=1)
    assert audio.stereo_stats(dual)["channel_correlation"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    independent = rng.standard_normal((SR, 2)).astype(np.float32)
    assert abs(audio.stereo_stats(independent)["channel_correlation"]) < 0.1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_probe_and_decode_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-ar", "48000", "-ac", "1", str(path)],
        check=True,
    )
    meta = audio.probe(path)
    assert meta["codec"] == "mp3" and meta["sample_rate"] == 48000 and meta["channels"] == 1
    samples = audio.decode(path, 16000, 1)
    assert samples.shape[1] == 1 and 0.9 < samples.shape[0] / 16000 < 1.2
