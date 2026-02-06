"""Pipecat bot with file-based audio transport for multi-turn eval.

Usage: python bot.py
"""

import asyncio
import glob
import os

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
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from file_audio_transport import FileAudioTransport

load_dotenv(override=True)


async def main():
    audio_files = sorted(glob.glob("audio_turns/*.wav"))
    if not audio_files:
        logger.error("No WAV files in audio_turns/. Add turn_01.wav, turn_02.wav, etc.")
        return

    logger.info(f"Found {len(audio_files)} audio turns: {audio_files}")

    transport = FileAudioTransport(
        audio_files=audio_files,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        save_bot_audio=True,
        output_dir="output_audio",
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="aura-asteria-en",
        encoding="linear16",
        sample_rate=24000,
    )
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": """
            You are a friendly AI assistant. Respond naturally and keep your answers conversational.

            Tell your name is Alan and you age is 25.
            """.strip(),
        },
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=ExternalUserTurnStrategies(),
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

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=True)

    async def _run():
        bot_task = asyncio.create_task(runner.run(task))
        await asyncio.sleep(1)
        await transport.run()
        await bot_task

    await _run()


if __name__ == "__main__":
    asyncio.run(main())
