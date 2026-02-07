import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.server import WebsocketServerParams, WebsocketServerTransport

from pipecat_compat import apply_broadcast_frame_instance_patch

load_dotenv(override=True)

SYSTEM_PROMPT = """
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.
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
    apply_broadcast_frame_instance_patch()

    host = os.getenv("PIPECAT_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("PIPECAT_AGENT_PORT", "8765"))

    transport = WebsocketServerTransport(
        host=host,
        port=port,
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en"),
        encoding="linear16",
        sample_rate=int(os.getenv("PIPECAT_AGENT_TTS_SAMPLE_RATE", "16000")),
    )
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("PIPECAT_AGENT_LLM_MODEL", "gpt-4o-mini"),
    )
    transcript_logger = TranscriptionLogger(side="AGENT")

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            transcript_logger,
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

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
