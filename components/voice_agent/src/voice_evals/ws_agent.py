from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, TextIO

import websockets

from voice_evals.audio_utils import chunk_audio, compute_bytes_per_chunk

logger = logging.getLogger(__name__)


@dataclass
class AgentTurnResult:
    assistant_text: str
    assistant_audio_bytes: bytes


def _truncate(value: str, max_len: int = 400) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}...[truncated {len(value) - max_len} chars]"


def _sanitize_event(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "audio" and isinstance(value, str):
            sanitized[key] = f"<base64:{len(value)} chars>"
        elif isinstance(value, str):
            sanitized[key] = _truncate(value)
        else:
            sanitized[key] = value
    return sanitized


class AgentWebSocketClient:
    def __init__(
        self,
        *,
        url: str,
        headers: Optional[dict[str, str]] = None,
        schema: str = "voice_agent_v1",
        input_mode: str = "raw_audio",
        output_mode: str = "json_events",
        tts_chunk_field: str = "audio",
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._schema = schema
        self._input_mode = input_mode
        self._output_mode = output_mode
        self._tts_chunk_field = tts_chunk_field
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._event_index = 0

    async def __aenter__(self) -> "AgentWebSocketClient":
        logger.info("Connecting to agent websocket: %s", self._url)
        self._ws = await websockets.connect(self._url, additional_headers=self._headers)
        logger.info("Agent websocket connected")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._ws:
            await self._ws.close()
            logger.info("Agent websocket closed")
        self._ws = None

    async def send_audio(
        self,
        audio_bytes: bytes,
        *,
        chunk_ms: int,
        sample_rate_hz: int,
        audio_format: str,
        real_time: bool,
    ) -> None:
        if self._input_mode != "raw_audio":
            raise ValueError(f"Unsupported input_mode: {self._input_mode}")
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        bytes_per_chunk = compute_bytes_per_chunk(sample_rate_hz, chunk_ms, audio_format)
        logger.debug(
            "Sending audio to agent: bytes=%s chunk_ms=%s bytes_per_chunk=%s format=%s real_time=%s",
            len(audio_bytes),
            chunk_ms,
            bytes_per_chunk,
            audio_format,
            real_time,
        )
        for chunk in chunk_audio(audio_bytes, bytes_per_chunk):
            await self._ws.send(chunk)
            if real_time:
                await asyncio.sleep(chunk_ms / 1000.0)

    async def collect_turn(
        self,
        *,
        idle_timeout_ms: int,
        log_file: Optional[TextIO] = None,
        turn_index: Optional[int] = None,
    ) -> AgentTurnResult:
        if self._output_mode != "json_events":
            raise ValueError(f"Unsupported output_mode: {self._output_mode}")
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        assistant_text_parts: list[str] = []
        assistant_audio = bytearray()
        idle_timeout = idle_timeout_ms / 1000.0
        turn_start = time.monotonic()
        last_event_time = turn_start

        while True:
            timeout = max(0.0, idle_timeout - (time.monotonic() - last_event_time))
            if timeout == 0:
                break
            try:
                message = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.debug(
                    "Turn %s: idle timeout reached after %sms",
                    turn_index,
                    idle_timeout_ms,
                )
                break

            last_event_time = time.monotonic()
            if isinstance(message, (bytes, bytearray)):
                # Unexpected binary frame; skip but keep listening.
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue

            self._event_index += 1
            if log_file is not None:
                record = {
                    "event_index": self._event_index,
                    "event_time_ms": int((time.monotonic() - turn_start) * 1000),
                    "turn_index": turn_index,
                    "event": _sanitize_event(payload),
                }
                log_file.write(json.dumps(record))
                log_file.write("\n")

            event_type = payload.get("type", "")
            logger.debug("Turn %s: received event type=%s", turn_index, event_type)
            if self._schema == "voice_agent_v1":
                if event_type == "agent_chunk":
                    text = payload.get("text", "")
                    if text:
                        assistant_text_parts.append(text)
                elif event_type == "tts_chunk":
                    audio_b64 = payload.get(self._tts_chunk_field)
                    if isinstance(audio_b64, str):
                        try:
                            assistant_audio.extend(base64.b64decode(audio_b64))
                        except (ValueError, TypeError):
                            pass
            else:
                # Unknown schema: just capture audio/text if explicit fields exist.
                text = payload.get("text")
                if isinstance(text, str):
                    assistant_text_parts.append(text)
                audio_b64 = payload.get(self._tts_chunk_field)
                if isinstance(audio_b64, str):
                    try:
                        assistant_audio.extend(base64.b64decode(audio_b64))
                    except (ValueError, TypeError):
                        pass

        return AgentTurnResult(
            assistant_text="".join(assistant_text_parts).strip(),
            assistant_audio_bytes=bytes(assistant_audio),
        )
