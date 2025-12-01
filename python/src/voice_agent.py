from typing import Any
import asyncio

from dotenv import load_dotenv
from typing_extensions import AsyncIterator
from langchain_core.runnables import RunnableGenerator
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
import pyaudio

from assemblyai_stt import microphone_and_transcribe
from elevenlabs_tts import text_to_speech_stream

load_dotenv()


agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[],
)


async def _stream_agent(
    input: AsyncIterator[tuple[AIMessage, Any]]
) -> AsyncIterator[str]:
    print("[DEBUG] _stream_agent: Starting agent stream")
    async for chunk in input:
        print(f"[DEBUG] _stream_agent: Received chunk: {chunk}")
        input_message = {"role": "user", "content": chunk}
        print(f"[DEBUG] _stream_agent: Sending to agent: {input_message}")
        async for message, _ in agent.astream({"messages": [input_message]}, stream_mode="messages"):
            print(f"[DEBUG] _stream_agent: Agent response: {message.text}")
            yield message.text


# ElevenLabs TTS - synthesize and play audio
async def _tts_stream(input: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    Convert text to speech using ElevenLabs and play through speakers.

    Args:
        input: AsyncIterator of text strings from agent

    Yields:
        Status messages (for pipeline continuity)
    """
    print("[DEBUG] _tts_stream: Starting TTS")

    # Initialize audio output
    p = pyaudio.PyAudio()
    audio_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        output=True,
        frames_per_buffer=1600
    )
    print("[DEBUG] Audio output stream opened")

    try:
        # Synthesize and play audio
        async for audio_chunk in text_to_speech_stream(input):
            # Play audio chunk through speakers
            await asyncio.get_event_loop().run_in_executor(
                None, audio_stream.write, audio_chunk
            )

        print("[DEBUG] _tts_stream: Finished playing audio")
        yield "tts_complete"

    finally:
        # Clean up audio
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()
        print("[DEBUG] Audio output closed")


audio_stream = (
    RunnableGenerator(microphone_and_transcribe)  # Combined mic + transcription
    | RunnableGenerator(_stream_agent)
    | RunnableGenerator(_tts_stream)
)


async def main():
    """
    Voice pipeline: Microphone → AssemblyAI STT → Agent → TTS
    """
    print("Starting voice pipeline...")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    try:
        print("[DEBUG] main: Starting audio_stream.astream(None)")
        async for output in audio_stream.astream(None):
            print(f"[DEBUG] main: Final output: {output}")
    except KeyboardInterrupt:
        print("\n\nStopping pipeline...")
    except Exception as e:
        print(f"[DEBUG] main: Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
