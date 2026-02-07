from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - python<3.11
    tomllib = None


@dataclass(frozen=True)
class RunConfig:
    name: str = ""
    results_dir: Path = Path("evals/results")
    max_turns: int = 6
    end_token: str = "END_CALL"


@dataclass(frozen=True)
class AudioConfig:
    input_format: str = "pcm16"
    input_sample_rate_hz: int = 16000
    output_format: str = "pcm16"
    output_sample_rate_hz: int = 24000
    chunk_ms: int = 20
    real_time: bool = True
    silence_padding_ms: int = 200
    pause_after_turn_ms: int = 300
    assistant_idle_timeout_ms: int = 800


@dataclass(frozen=True)
class AgentWsConfig:
    url: str = "ws://localhost:8000/ws"
    schema: str = "voice_agent_v1"
    input_mode: str = "raw_audio"
    output_mode: str = "json_events"
    tts_chunk_field: str = "audio"
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSConfig:
    provider: str = "cartesia"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class STTConfig:
    provider: str = "deepgram"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulatorConfig:
    mode: str = "stt_model_tts"
    system_prompt: str = ""
    start_prompt: str = "Begin the conversation as the user."
    model: str = "openai:gpt-4o-mini"
    voice: str = ""
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    realtime: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalConfig:
    run: RunConfig = field(default_factory=RunConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    agent_ws: AgentWsConfig = field(default_factory=AgentWsConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)


def _coerce_path(value: Any, default: Path) -> Path:
    if value is None or value == "":
        return default
    return Path(value)


def _load_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".toml", ".tml"}:
        if tomllib is None:
            raise RuntimeError("tomllib not available; use .json config instead.")
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError("Unsupported config format. Use .json or .toml.")


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_run(data: Dict[str, Any]) -> RunConfig:
    return RunConfig(
        name=str(data.get("name", "")),
        results_dir=_coerce_path(data.get("results_dir"), Path("evals/results")),
        max_turns=int(data.get("max_turns", 6)),
        end_token=str(data.get("end_token", "END_CALL")),
    )


def _build_audio(data: Dict[str, Any]) -> AudioConfig:
    return AudioConfig(
        input_format=str(data.get("input_format", "pcm16")),
        input_sample_rate_hz=int(data.get("input_sample_rate_hz", 16000)),
        output_format=str(data.get("output_format", "pcm16")),
        output_sample_rate_hz=int(data.get("output_sample_rate_hz", 24000)),
        chunk_ms=int(data.get("chunk_ms", 20)),
        real_time=bool(data.get("real_time", True)),
        silence_padding_ms=int(data.get("silence_padding_ms", 200)),
        pause_after_turn_ms=int(data.get("pause_after_turn_ms", 300)),
        assistant_idle_timeout_ms=int(data.get("assistant_idle_timeout_ms", 800)),
    )


def _build_agent_ws(data: Dict[str, Any]) -> AgentWsConfig:
    headers = data.get("headers", {}) or {}
    if not isinstance(headers, dict):
        raise ValueError("agent_ws.headers must be a map of string->string")
    return AgentWsConfig(
        url=str(data.get("url", "ws://localhost:8000/ws")),
        schema=str(data.get("schema", "voice_agent_v1")),
        input_mode=str(data.get("input_mode", "raw_audio")),
        output_mode=str(data.get("output_mode", "json_events")),
        tts_chunk_field=str(data.get("tts_chunk_field", "audio")),
        headers={str(k): str(v) for k, v in headers.items()},
    )


def _build_tts(data: Optional[Dict[str, Any]]) -> TTSConfig:
    if not data:
        return TTSConfig()
    return TTSConfig(
        provider=str(data.get("provider", "cartesia")),
        options=dict(data.get("options", {}) or {}),
    )


def _build_stt(data: Optional[Dict[str, Any]]) -> STTConfig:
    if not data:
        return STTConfig()
    return STTConfig(
        provider=str(data.get("provider", "deepgram")),
        options=dict(data.get("options", {}) or {}),
    )


def _build_simulator(data: Dict[str, Any]) -> SimulatorConfig:
    return SimulatorConfig(
        mode=str(data.get("mode", "stt_model_tts")),
        system_prompt=str(data.get("system_prompt", "")),
        start_prompt=str(data.get("start_prompt", "Begin the conversation as the user.")),
        model=str(data.get("model", "openai:gpt-4o-mini")),
        voice=str(data.get("voice", "")),
        tts=_build_tts(data.get("tts")),
        stt=_build_stt(data.get("stt")),
        realtime=dict(data.get("realtime", {}) or {}),
    )


def load_config(path: Path, overrides: Optional[Dict[str, Any]] = None) -> EvalConfig:
    raw = _load_dict(path)
    if overrides:
        raw = _merge_dict(raw, overrides)

    return EvalConfig(
        run=_build_run(raw.get("run", {})),
        audio=_build_audio(raw.get("audio", {})),
        agent_ws=_build_agent_ws(raw.get("agent_ws", {})),
        simulator=_build_simulator(raw.get("simulator", {})),
    )
