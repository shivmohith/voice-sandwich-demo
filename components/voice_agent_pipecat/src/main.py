import asyncio
import os

import aiohttp
from deepgram import LiveOptions
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.server import WebsocketServerParams, WebsocketServerTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.turns.user_stop import TranscriptionUserTurnStopStrategy, TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

load_dotenv(override=True)

SYSTEM_PROMPT = """
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.

KEEP RESPONSES SHORT AND CONCISE.
""".strip()

class TranscriptionLogger(FrameProcessor):
    def __init__(self, side: str):
        super().__init__()
        self._side = side

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.info(f"[{self._side} STT interim] {text}")
        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.info(f"[{self._side} STT final] {text}")

        await self.push_frame(frame, direction)


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        using_smart_turn_detection = True
        host = os.getenv("PIPECAT_AGENT_HOST", "0.0.0.0")
        port = int(os.getenv("PIPECAT_AGENT_PORT", "8765"))
        input_sample_rate = int(os.getenv("PIPECAT_AGENT_INPUT_SAMPLE_RATE", "16000"))
        output_sample_rate = int(os.getenv("PIPECAT_AGENT_TTS_SAMPLE_RATE", "16000"))

        transport = WebsocketServerTransport(
            host=host,
            port=port,
            params=WebsocketServerParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=ProtobufFrameSerializer(),
                session_timeout=60 * 3,  # 3 minutes
            ),
        )

        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            sample_rate=input_sample_rate,
            live_options=LiveOptions(
                encoding="linear16",
                channels=1,
                model=os.getenv("DEEPGRAM_STT_MODEL", "nova-3-general"),
                interim_results=False,
                # smart_format=True,
                # punctuate=True,
                # profanity_filter=True,
                # # Needed so Deepgram emits utterance boundaries while stream stays open.
                # endpointing=int(os.getenv("DEEPGRAM_STT_ENDPOINTING_MS", "500")),
                # utterance_end_ms=str(int(os.getenv("DEEPGRAM_STT_UTTERANCE_END_MS", "1000"))),
                # vad_events=True,
            ),
        )

        # stt = ElevenLabsSTTService(
        #     api_key=os.getenv("ELEVEN_API_KEY"),
        #     model="scribe_v2",
        #     sample_rate=16000,
        #     aiohttp_session=session,
        # )
        
        llm = GeminiLiveLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
            system_instruction=SYSTEM_PROMPT,
        )

        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en"),
            # encoding="linear16",
            sample_rate=output_sample_rate,
        )
        # llm = OpenAILLMService(
        #     api_key=os.getenv("OPENAI_API_KEY"),
        #     model=os.getenv("PIPECAT_AGENT_LLM_MODEL", "gpt-4o-mini"),
        # )
        transcript_logger = TranscriptionLogger(side="AGENT")

        context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
        # user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        #     context,
        #     user_params=LLMUserAggregatorParams(
        #         # vad_analyzer=vad_analyzer,
        #         user_turn_strategies=UserTurnStrategies(
        #             stop=[TurnAnalyzerUserTurnStopStrategy(
        #                 turn_analyzer=LocalSmartTurnAnalyzerV3()
        #             )]
        #         ) if using_smart_turn_detection else TranscriptionUserTurnStopStrategy(),
        #     ),
        # )

        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                # user_turn_strategies=ExternalUserTurnStrategies(),
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            ),
        )

        pipeline = Pipeline(
            [
                transport.input(),
                # stt,
                # transcript_logger,
                user_aggregator,
                llm,
                # tts,
                transport.output(),
                assistant_aggregator,
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
                audio_in_sample_rate=input_sample_rate,
                audio_out_sample_rate=output_sample_rate,
            ),
            enable_rtvi=False,
        )

        @transport.event_handler("on_websocket_ready")
        async def on_websocket_ready(_transport):
            logger.info(f"Pipecat voice agent listening on ws://{host}:{port}")

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, client):
            logger.info(f"Client connected: {client.remote_address}")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, client):
            logger.info(f"Client disconnected: {getattr(client, 'remote_address', None)}")
            await task.cancel()

        @transport.event_handler("on_session_timeout")
        async def on_session_timeout(transport, client):
            logger.info(f"Entering in timeout for {client.remote_address}")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=True)
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
