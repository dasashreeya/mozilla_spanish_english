"""Diagnose a GPU decode failure without touching any competition data.

    uv run --extra train python scripts/modal_probe.py-via-watchdog:
    uv run --extra train python scripts/modal_watchdog.py probe <run-id> \\
        --script scripts/modal_probe.py --max-minutes 15

Why this file exists, separately from scripts/modal_decode.py: ger-smoke2 died inside the decode
loop, where every exception is scrubbed to its type name so a traceback can never name a clip.
That is the right rule for a run that reads the corpus, and it made the failure undiagnosable.

This module reads NO manifest, NO audio file and NO dataset directory. Its only input is a numpy
array of zeros. There is therefore nothing here that could name competition data, which is why it
prints tracebacks in full and does not silence library logging -- for a probe, the chatter is the
evidence. Keeping it in its own file is what lets modal_decode.py stay under its surface lock.
"""

from __future__ import annotations

import modal

app = modal.App("lit-sp-en-probe")
volume = modal.Volume.from_name("lit-sp-en-train")

# Deliberately self-contained: Modal ships only the entrypoint file into the container, so a
# `from modal_decode import ...` resolves locally and then dies inside the container with
# ModuleNotFoundError -- which is exactly how ger-probe1 failed, after ~8 minutes of image
# build. scripts/modal_finetune.py avoids this with `.add_local_python_source(...)`; a probe
# should not depend on a second file at all. These values are duplicated from
# scripts/modal_decode.py; tests/test_modal_probe.py asserts they stay identical.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("faster-whisper==1.2.1", "ctranslate2==4.8.2", "soundfile==0.14.0",
                 "numpy==2.3.5", "huggingface-hub==0.36.2", "tokenizers==0.22.2",
                 "nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9")
    .run_commands(
        # `find`, not a Python import: nvidia.cublas.lib is a NAMESPACE package, so
        # `__file__` is None and os.path.dirname(None) raises -- which is how the
        # first version of this step failed the build.
        "find /usr/local/lib/python3.12/site-packages/nvidia -name '*.so*' -printf '%h\\n' | sort -u > /etc/ld.so.conf.d/nvidia.conf",
        "ldconfig",
        "ldconfig -p | grep -c libcublas.so.12",   # fails the build if it is still missing
    )
)
GPU = "A100-80GB"
CPU = (8.0, 8.0)
MEMORY = (65536, 65536)
VOLUME = "/vol"
STARTUP_TIMEOUT = 10 * 60
TURBO_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
VOTERS = {
    "turbo": {"source": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "on_volume": False,
              "prompted": True, "revision": TURBO_REVISION},
}


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=20 * 60,
              startup_timeout=STARTUP_TIMEOUT, volumes={VOLUME: volume}, retries=0)
def probe(run_id: str = "probe") -> dict:
    """Load the turbo voter and transcribe one second of synthetic silence.

    ger-smoke2 died inside the decode loop, where every exception is scrubbed to its type name
    so a traceback cannot name a clip. That made the run undiagnosable. This function touches no
    manifest, no audio file and no dataset directory - the only input is a numpy array of zeros -
    so there is nothing here that could name competition data, and errors are printed in full.

    It also does NOT call _silence_libraries(): for a probe, library chatter is the evidence.
    """
    import traceback

    import numpy as np

    from faster_whisper import WhisperModel

    spec = VOTERS["turbo"]
    print(f"PROBE start run={run_id} source={spec['source']} revision={spec['revision']}",
          flush=True)
    try:
        import ctranslate2

        print(f"PROBE ctranslate2={ctranslate2.__version__} "
              f"cuda_device_count={ctranslate2.get_cuda_device_count()}", flush=True)
    except Exception:
        print("PROBE ctranslate2 import/inspect failed:", flush=True)
        traceback.print_exc()

    try:
        model = WhisperModel(spec["source"], device="cuda", compute_type="float16",
                             revision=spec["revision"])
        print("PROBE model loaded", flush=True)
    except Exception:
        print("PROBE model load FAILED:", flush=True)
        traceback.print_exc()
        return {"probe": "model-load-failed"}

    try:
        silence = np.zeros(16000, dtype=np.float32)  # 1 s at 16 kHz, synthetic
        segments, info = model.transcribe(silence, language=None, task="transcribe", beam_size=5)
        n = len(list(segments))
        print(f"PROBE transcribe OK segments={n} detected_language={info.language}", flush=True)
        return {"probe": "ok", "segments": n}
    except Exception:
        print("PROBE transcribe FAILED:", flush=True)
        traceback.print_exc()
        return {"probe": "transcribe-failed"}
