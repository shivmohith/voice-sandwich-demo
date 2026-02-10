"""Pipecat bot with pluggable audio transports for multi-turn eval.

Supports two transport modes (set via TRANSPORT env var):
  - "file"      (default): Reads user audio from WAV files in audio_turns/
  - "websocket": Connects to an external voice agent as a simulated user

Usage:
    # File transport (default)
    python bot.py

    # WebSocket transport (simulated user talking to external agent)
    TRANSPORT=websocket python bot.py
"""

import asyncio
import glob
import os
from pickle import TRUE

import aiohttp
from deepgram import LiveOptions
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.turns.user_turn_strategies import TranscriptionUserTurnStopStrategy, TranscriptionUserTurnStartStrategy, UserTurnStrategies
from pipecat.services.deepgram.tts import DeepgramTTSService

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams

load_dotenv(override=True)

# System prompt used when the bot is a self-contained assistant (file transport)
FILE_SYSTEM_PROMPT = """
You are a friendly AI assistant. Respond naturally and keep your answers conversational.

Tell your name is Alan and you age is 25.
""".strip()

# System prompt used when the bot acts as a simulated customer (websocket transport)
WS_SYSTEM_PROMPT = """
You are a customer at a sandwich shop. You are placing an order via voice.
Be natural and conversational. Order a turkey sandwich with lettuce, tomato,
and swiss cheese. If the agent asks for confirmation, confirm your order.
Keep your responses short (1-2 sentences).
""".strip()


def _create_file_transport():
    from file_audio_transport import FileAudioTransport

    audio_files = sorted(glob.glob("audio_turns/input_math_1_e11labs.wav"))
    if not audio_files:
        logger.error("No WAV files in audio_turns/. Add turn_01.wav, turn_02.wav, etc.")
        return None

    logger.info(f"Found {len(audio_files)} audio turns: {audio_files}")

    return FileAudioTransport(
        audio_files=audio_files,
        params=TransportParams(
            audio_in_enabled=True, 
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
        ),
        save_bot_audio=True,
        output_dir="output_audio",
    )


def _create_ws_transport():
    from ws_agent_transport import WSAgentTransport

    uri = os.getenv("AGENT_WS_URL", "ws://localhost:8000/ws")
    max_turns = min(int(os.getenv("MAX_TURNS", "5")), 5)

    logger.info(f"WebSocket transport: uri={uri}, max_turns={max_turns}")

    return WSAgentTransport(
        uri=uri,
        params=TransportParams(
            audio_in_enabled=True, audio_out_enabled=True),
        max_turns=max_turns,
        save_audio=True,
        output_dir="simulation_audio",
    )


async def main():
    async with aiohttp.ClientSession() as session:
        transport_mode = os.getenv("TRANSPORT", "file").lower()
        logger.info(f"Transport mode: {transport_mode}")

        if transport_mode == "websocket":
            transport = _create_ws_transport()
            system_prompt = WS_SYSTEM_PROMPT
        else:
            transport = _create_file_transport()
            system_prompt = FILE_SYSTEM_PROMPT

        if transport is None:
            return

        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"), 
            sample_rate=16000,
            live_options=LiveOptions(
                encoding="linear16",
                channels=1,
                model="nova-3-general",
                interim_results=True,
                # smart_format=True,
                # punctuate=True,
                # profanity_filter=True,
            ),
        )

        # stt = ElevenLabsSTTService(
        #     api_key=os.getenv("ELEVEN_API_KEY"),
        #     model="scribe_v2",
        #     sample_rate=16000,
        #     aiohttp_session=session,
        # )

        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model="aura-asteria-en",
            encoding="linear16",
            sample_rate=24000,
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

        messages = [{"role": "system", "content": system_prompt}]
        context = LLMContext(messages)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                # user_turn_strategies=ExternalUserTurnStrategies(),
                # user_turn_strategies=UserTurnStrategies(
                #     start=[TranscriptionUserTurnStartStrategy()],
                #     stop=[TranscriptionUserTurnStopStrategy()],
                # ),
            ),
        )

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                user_aggregator,
                llm,
                tts,
                transport.output(),
                assistant_aggregator,
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            if transport_mode == "websocket":
                # Simulated user speaks first — trigger the LLM to generate the opening message
                await task.queue_frame(LLMRunFrame())

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=True)

        async def _run():
            bot_task = asyncio.create_task(runner.run(task))
            await asyncio.sleep(1)
            await transport.run()
            await bot_task  # Wait for pipeline to fully stop
            if hasattr(transport, "close"):
                await transport.close()

        await _run()


if __name__ == "__main__":
    asyncio.run(main())
