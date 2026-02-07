"""Pipecat user simulator that talks to a Pipecat WebSocket voice agent."""

import asyncio
import os
import time
import wave
from datetime import datetime
from typing import Awaitable, Callable, Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMRunFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_compat import apply_broadcast_frame_instance_patch
from ws_agent_transport_pipecat import PipecatWSAgentTransport

load_dotenv(override=True)

WS_SYSTEM_PROMPT = """
You are a customer at a sandwich shop. You are placing an order via voice.
Be natural and conversational. Order a turkey sandwich with lettuce, tomato,
and swiss cheese. If the agent asks for confirmation, confirm your order.
Keep your responses short (1-2 sentences).
""".strip()


class TranscriptionLogger(FrameProcessor):
    def __init__(self, side: str):
        super().__init__()
        self._side = side
        self._on_activity: Optional[Callable[[], None]] = None

    def set_activity_callback(self, callback: Callable[[], None]):
        self._on_activity = callback

    def _mark_activity(self):
        if self._on_activity:
            self._on_activity()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            self._mark_activity()
            text = frame.text.strip()
            if text:
                logger.info(f"[{self._side} STT interim] {text}")
        elif isinstance(frame, TranscriptionFrame):
            self._mark_activity()
            text = frame.text.strip()
            if text:
                logger.info(f"[{self._side} STT final] {text}")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._mark_activity()

        await self.push_frame(frame, direction)


class SimulationAudioRecorder:
    def __init__(self, output_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self._output_dir, exist_ok=True)

        self._agent_audio_chunks: list[bytes] = []
        self._agent_sample_rate = 16000
        self._sim_user_audio_chunks: list[bytes] = []
        self._sim_user_sample_rate = 16000
        self._exchange_turn = 0

        logger.info(f"Saving simulation audio to: {self._output_dir}")

    def append_agent_audio(self, frame: InputAudioRawFrame):
        if frame.audio:
            self._agent_audio_chunks.append(frame.audio)
            self._agent_sample_rate = frame.sample_rate

    def append_sim_user_audio(self, frame: OutputAudioRawFrame):
        if frame.audio:
            self._sim_user_audio_chunks.append(frame.audio)
            self._sim_user_sample_rate = frame.sample_rate

    def _save_wav(self, filename: str, audio_data: bytes, sample_rate: int):
        path = os.path.join(self._output_dir, filename)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        logger.info(f"Saved audio: {path}")

    def flush_agent_for_current_exchange(self) -> Optional[int]:
        if not self._agent_audio_chunks or self._exchange_turn <= 0:
            return None

        audio_data = b"".join(self._agent_audio_chunks)
        self._agent_audio_chunks.clear()
        self._save_wav(
            f"turn_{self._exchange_turn:03d}_agent.wav",
            audio_data,
            self._agent_sample_rate,
        )
        return self._exchange_turn

    def complete_user_turn(self) -> Optional[int]:
        if not self._sim_user_audio_chunks:
            return None

        self._exchange_turn += 1
        audio_data = b"".join(self._sim_user_audio_chunks)
        self._sim_user_audio_chunks.clear()
        self._save_wav(
            f"turn_{self._exchange_turn:03d}_simulated_user.wav",
            audio_data,
            self._sim_user_sample_rate,
        )
        return self._exchange_turn

    def flush_all(self):
        if self._sim_user_audio_chunks:
            self.complete_user_turn()
        self.flush_agent_for_current_exchange()


class IncomingAudioRecorder(FrameProcessor):
    def __init__(self, recorder: SimulationAudioRecorder):
        super().__init__()
        self._recorder = recorder
        self._on_activity: Optional[Callable[[], None]] = None

    def set_activity_callback(self, callback: Callable[[], None]):
        self._on_activity = callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._recorder.append_agent_audio(frame)
            if self._on_activity:
                self._on_activity()

        await self.push_frame(frame, direction)


class OutgoingAudioRecorder(FrameProcessor):
    def __init__(self, recorder: SimulationAudioRecorder):
        super().__init__()
        self._recorder = recorder
        self._on_turn_complete: Optional[Callable[[int], Awaitable[None]]] = None
        self._on_activity: Optional[Callable[[], None]] = None

    def set_turn_complete_callback(self, callback: Callable[[int], Awaitable[None]]):
        self._on_turn_complete = callback

    def set_activity_callback(self, callback: Callable[[], None]):
        self._on_activity = callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame):
            self._recorder.append_sim_user_audio(frame)
            if self._on_activity:
                self._on_activity()
        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._on_activity:
                self._on_activity()
            self._recorder.flush_agent_for_current_exchange()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._on_activity:
                self._on_activity()
            turn = self._recorder.complete_user_turn()
            if turn is not None and self._on_turn_complete:
                await self._on_turn_complete(turn)

        await self.push_frame(frame, direction)


async def main():
    apply_broadcast_frame_instance_patch()

    uri = os.getenv("AGENT_WS_URL", "ws://localhost:8765")
    max_turns = int(os.getenv("MAX_TURNS", "2"))
    idle_timeout_secs = float(os.getenv("SIMULATOR_IDLE_TIMEOUT_SECS", "20"))
    simulation_audio_dir = os.getenv("SIMULATION_AUDIO_DIR", "simulation_audio")

    logger.info(f"Pipecat simulator connecting to: {uri}")
    logger.info(
        f"Simulator limits: max_turns={max_turns} idle_timeout_secs={idle_timeout_secs}"
    )

    transport = PipecatWSAgentTransport(uri=uri)
    audio_recorder = SimulationAudioRecorder(simulation_audio_dir)

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="aura-asteria-en",
        encoding="linear16",
        sample_rate=16000,
    )
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    incoming_audio_recorder = IncomingAudioRecorder(audio_recorder)
    transcript_logger = TranscriptionLogger(side="SIMULATOR")
    outgoing_audio_recorder = OutgoingAudioRecorder(audio_recorder)

    context = LLMContext([{"role": "system", "content": WS_SYSTEM_PROMPT}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            incoming_audio_recorder,
            stt,
            transcript_logger,
            user_aggregator,
            llm,
            tts,
            outgoing_audio_recorder,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        enable_rtvi=False,
    )

    shutdown_started = False
    last_activity_ts = time.monotonic()

    def mark_activity():
        nonlocal last_activity_ts
        last_activity_ts = time.monotonic()

    async def stop_simulation(reason: str):
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        logger.info(f"Stopping simulator: {reason}")
        await task.cancel()
        await transport.stop()

    transcript_logger.set_activity_callback(mark_activity)
    incoming_audio_recorder.set_activity_callback(mark_activity)
    outgoing_audio_recorder.set_activity_callback(mark_activity)

    async def on_sim_user_turn_completed(turn_count: int):
        logger.info(f"[SIMULATOR TURN] completed={turn_count}/{max_turns}")
        if max_turns > 0 and turn_count >= max_turns:
            await stop_simulation(f"max turns reached ({turn_count})")

    outgoing_audio_recorder.set_turn_complete_callback(on_sim_user_turn_completed)

    @transport.event_handler("on_connected")
    async def on_connected(_transport, _client):
        logger.info("Connected to external Pipecat agent")
        mark_activity()
        await task.queue_frame(LLMRunFrame())

    @transport.event_handler("on_disconnected")
    async def on_disconnected(_transport, _client):
        logger.info("Disconnected from external Pipecat agent")
        await stop_simulation("transport disconnected")

    runner = PipelineRunner(handle_sigint=True)

    async def idle_watchdog():
        if idle_timeout_secs <= 0:
            return
        while not shutdown_started:
            await asyncio.sleep(1.0)
            idle_for = time.monotonic() - last_activity_ts
            if idle_for >= idle_timeout_secs:
                await stop_simulation(f"idle timeout reached ({idle_for:.1f}s)")
                return

    async def _run():
        bot_task = asyncio.create_task(runner.run(task))
        watchdog_task = asyncio.create_task(idle_watchdog())

        await transport.run()

        watchdog_task.cancel()
        await asyncio.gather(watchdog_task, return_exceptions=True)
        await bot_task

        audio_recorder.flush_all()
        await transport.close()

    await _run()


if __name__ == "__main__":
    asyncio.run(main())
