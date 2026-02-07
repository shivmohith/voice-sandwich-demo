from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from assemblyai_stt import AssemblyAISTT
from deepgram_stt import DeepgramSTT
from events import STTOutputEvent
from voice_evals.audio_utils import chunk_audio, compute_bytes_per_chunk
from voice_evals.config import AudioConfig, SimulatorConfig
from deepgram_tts import DeepgramTTS

logger = logging.getLogger(__name__)


@dataclass
class SimulatorTurn:
    text: str
    audio_bytes: bytes
    should_end: bool


class UserSimulator:
    async def next_turn(
        self, last_assistant_text: Optional[str], turn_index: int
    ) -> SimulatorTurn:
        raise NotImplementedError


class LangChainTextGenerator:
    def __init__(self, *, system_prompt: str, model: str) -> None:
        self._agent = create_agent(
            model=model,
            tools=[],
            system_prompt=system_prompt,
            checkpointer=InMemorySaver(),
        )
        self._thread_id = str(uuid4())

    async def generate(self, prompt: str) -> str:
        logger.debug("Simulator text generation prompt: %s", prompt)
        stream = self._agent.astream(
            {"messages": [HumanMessage(content=prompt)]},
            {"configurable": {"thread_id": self._thread_id}},
            stream_mode="messages",
        )
        chunks: list[str] = []
        async for message, _metadata in stream:
            if isinstance(message, AIMessage):
                if message.text:
                    chunks.append(message.text)
        return "".join(chunks).strip()


class TTSAdapter:
    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError


DEFAULT_TTS_IDLE_TIMEOUT_MS = 1200


async def _collect_tts_audio(tts, *, idle_timeout_ms: Optional[int] = None) -> bytes:
    audio = bytearray()
    iterator = tts.receive_events().__aiter__()
    effective_timeout_ms = (
        idle_timeout_ms
        if idle_timeout_ms is not None
        else DEFAULT_TTS_IDLE_TIMEOUT_MS
    )
    logger.debug("Collecting TTS audio with idle_timeout_ms=%s", effective_timeout_ms)
    while True:
        try:
            event = await asyncio.wait_for(
                iterator.__anext__(), timeout=effective_timeout_ms / 1000.0
            )
        except StopAsyncIteration:
            logger.debug("TTS iterator completed")
            break
        except asyncio.TimeoutError:
            logger.debug("TTS audio collection timed out")
            break
        audio.extend(event.audio)
    logger.debug("Collected TTS audio bytes=%s", len(audio))
    await tts.close()
    return bytes(audio)


class DeepgramTTSAdapter(TTSAdapter):
    def __init__(self, *, idle_timeout_ms: int = 800, **options: Any) -> None:
        self._options = options
        self._idle_timeout_ms = idle_timeout_ms

    async def synthesize(self, text: str) -> bytes:
        tts = DeepgramTTS(**self._options)
        await tts.send_text(text)
        return await _collect_tts_audio(tts, idle_timeout_ms=self._idle_timeout_ms)


class SttModelTtsSimulator(UserSimulator):
    def __init__(
        self,
        *,
        text_generator: LangChainTextGenerator,
        tts: TTSAdapter,
        start_prompt: str,
        end_token: str,
    ) -> None:
        self._text_generator = text_generator
        self._tts = tts
        self._start_prompt = start_prompt
        self._end_token = end_token

    async def next_turn(
        self, last_assistant_text: Optional[str], turn_index: int
    ) -> SimulatorTurn:
        if turn_index == 1 or not last_assistant_text:
            prompt = self._start_prompt
        else:
            prompt = last_assistant_text
        user_text = await self._text_generator.generate(prompt)
        should_end = self._end_token.lower() in user_text.lower()
        logger.debug(
            "Simulator turn=%s text='%s' should_end=%s",
            turn_index,
            user_text,
            should_end,
        )
        audio_bytes = await self._tts.synthesize(user_text)
        return SimulatorTurn(text=user_text, audio_bytes=audio_bytes, should_end=should_end)


def _load_adapter(spec: str, options: dict[str, Any]) -> Any:
    if ":" not in spec:
        raise ValueError("Adapter spec must be in module:Class form")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, attr, None)
    if adapter_cls is None:
        raise ValueError(f"Adapter {attr} not found in {module_name}")
    return adapter_cls(**options)


def _build_tts_adapter(
    config: SimulatorConfig,
    audio: AudioConfig,
) -> TTSAdapter:
    provider = config.tts.provider.lower()
    options = dict(config.tts.options)
    if provider == "deepgram":
        # Accept Cartesia-style aliases in config and normalize for Deepgram.
        alias_encoding = str(options.get("encoding", "")).strip().lower()
        if alias_encoding == "pcm_s16le":
            options["encoding"] = "linear16"
        options.setdefault("encoding", "linear16")
        # Deepgram voice models commonly expect 24kHz output.
        options.setdefault("sample_rate", 24000)
        return DeepgramTTSAdapter(**options)
    if provider == "custom":
        adapter_spec = options.get("adapter")
        if not adapter_spec:
            raise ValueError("custom tts provider requires options.adapter")
        return _load_adapter(adapter_spec, options.get("init", {}))
    raise ValueError(f"Unsupported TTS provider: {provider}")


def create_simulator(
    *,
    simulator_config: SimulatorConfig,
    audio_config: AudioConfig,
    end_token: str,
) -> UserSimulator:
    mode = simulator_config.mode.lower()
    if mode == "stt_model_tts":
        text_generator = LangChainTextGenerator(
            system_prompt=simulator_config.system_prompt,
            model=simulator_config.model,
        )
        tts_adapter = _build_tts_adapter(simulator_config, audio_config)
        return SttModelTtsSimulator(
            text_generator=text_generator,
            tts=tts_adapter,
            start_prompt=simulator_config.start_prompt,
            end_token=end_token,
        )
    if mode == "realtime":
        realtime_cfg = simulator_config.realtime
        adapter_spec = realtime_cfg.get("adapter")
        if not adapter_spec:
            raise ValueError("Realtime simulator requires simulator.realtime.adapter")
        adapter = _load_adapter(adapter_spec, realtime_cfg.get("options", {}))
        return adapter
    raise ValueError(f"Unsupported simulator mode: {mode}")


async def transcribe_audio(
    *,
    audio_bytes: bytes,
    provider: str,
    sample_rate_hz: int,
    audio_format: str,
    chunk_ms: int,
) -> str:
    logger.debug(
        "Transcribing audio bytes=%s provider=%s sample_rate=%s format=%s chunk_ms=%s",
        len(audio_bytes),
        provider,
        sample_rate_hz,
        audio_format,
        chunk_ms,
    )
    if audio_format != "pcm16":
        raise ValueError("STT transcription currently requires pcm16 audio")

    if provider == "deepgram":
        stt = DeepgramSTT(sample_rate=sample_rate_hz)
    elif provider == "assemblyai":
        stt = AssemblyAISTT(sample_rate=sample_rate_hz)
    else:
        raise ValueError(f"Unsupported STT provider: {provider}")

    bytes_per_chunk = compute_bytes_per_chunk(sample_rate_hz, chunk_ms, audio_format)

    async def send_audio() -> None:
        for chunk in chunk_audio(audio_bytes, bytes_per_chunk):
            await stt.send_audio(chunk)
        await stt.close()

    sender = asyncio.create_task(send_audio())
    final_text = ""
    async for event in stt.receive_events():
        if isinstance(event, STTOutputEvent):
            final_text = event.transcript
            logger.debug("Received STT output transcript: %s", final_text)
    await sender
    await stt.close()
    logger.debug("Final STT transcript: %s", final_text.strip())
    return final_text.strip()
