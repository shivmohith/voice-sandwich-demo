from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_FORMATS = {"pcm16", "g711_ulaw", "g711_alaw"}


def bytes_per_sample(audio_format: str) -> int:
    if audio_format == "pcm16":
        return 2
    if audio_format in {"g711_ulaw", "g711_alaw"}:
        return 1
    raise ValueError(f"Unsupported audio format: {audio_format}")


def compute_bytes_per_chunk(
    sample_rate_hz: int, chunk_ms: int, audio_format: str
) -> int:
    bps = bytes_per_sample(audio_format)
    computed = int(sample_rate_hz * (chunk_ms / 1000.0) * bps)
    return max(1, computed)


def chunk_audio(audio_bytes: bytes, bytes_per_chunk: int) -> list[bytes]:
    return [
        audio_bytes[offset : offset + bytes_per_chunk]
        for offset in range(0, len(audio_bytes), bytes_per_chunk)
    ]


def silence_bytes(audio_format: str, duration_ms: int, sample_rate_hz: int) -> bytes:
    if duration_ms <= 0:
        return b""
    total_bytes = int(
        sample_rate_hz * (duration_ms / 1000.0) * bytes_per_sample(audio_format)
    )
    if audio_format == "pcm16":
        return b"\x00" * total_bytes
    if audio_format == "g711_ulaw":
        return b"\xFF" * total_bytes
    if audio_format == "g711_alaw":
        return b"\xD5" * total_bytes
    raise ValueError(f"Unsupported audio format: {audio_format}")


def write_pcm16_wav(path: Path, pcm_bytes: bytes, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm_bytes)


@dataclass(frozen=True)
class AudioArtifact:
    raw_path: Path
    meta_path: Path
    wav_path: Path | None


def write_audio_artifact(
    directory: Path,
    name: str,
    audio_bytes: bytes,
    audio_format: str,
    sample_rate_hz: int,
) -> AudioArtifact:
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / f"{name}.{audio_format}.raw"
    meta_path = directory / f"{name}.json"
    raw_path.write_bytes(audio_bytes)
    meta_path.write_text(
        json.dumps(
            {
                "format": audio_format,
                "sample_rate_hz": sample_rate_hz,
                "bytes": len(audio_bytes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    wav_path = None
    if audio_format == "pcm16":
        wav_path = directory / f"{name}.wav"
        write_pcm16_wav(wav_path, audio_bytes, sample_rate_hz)
    return AudioArtifact(raw_path=raw_path, meta_path=meta_path, wav_path=wav_path)
