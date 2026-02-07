"""Custom file-based audio transport for Pipecat.

Mocks a multi-turn conversation by reading user audio from WAV files
and feeding them into the pipeline. Each file = one user turn.

Usage:
    transport = FileAudioTransport(
        audio_files=["turn1.wav", "turn2.wav", "turn3.wav"],
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
"""

import asyncio
import struct
import wave
from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams


# How many milliseconds of audio to send per chunk (simulates real-time streaming)
CHUNK_DURATION_MS = 20


def read_wav_as_pcm16(filepath: str, target_sample_rate: int = 16000) -> tuple[bytes, int, int]:
    """Read a WAV file and return raw PCM16 bytes, sample_rate, num_channels."""
    with wave.open(filepath, "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw_data = wf.readframes(wf.getnframes())

    # Convert to 16-bit if needed
    if sample_width == 1:
        # 8-bit unsigned -> 16-bit signed
        samples = struct.unpack(f"{len(raw_data)}B", raw_data)
        raw_data = struct.pack(f"{len(samples)}h", *((s - 128) * 256 for s in samples))
    elif sample_width == 4:
        # 32-bit -> 16-bit (take upper 16 bits)
        samples = struct.unpack(f"{len(raw_data) // 4}i", raw_data)
        raw_data = struct.pack(f"{len(samples)}h", *(s >> 16 for s in samples))

    # Mono-ize if stereo
    if num_channels == 2:
        samples = struct.unpack(f"{len(raw_data) // 2}h", raw_data)
        mono = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
        raw_data = struct.pack(f"{len(mono)}h", *mono)
        num_channels = 1

    return raw_data, sample_rate, num_channels


class FileAudioInputTransport(BaseInputTransport):
    """Reads audio from WAV files and pushes frames into the pipeline."""

    def __init__(
        self,
        audio_files: List[str],
        params: TransportParams,
        bot_done_event: asyncio.Event,
        on_all_turns_done: asyncio.Event,
        inter_turn_pause_secs: float = 1.0,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._audio_files = audio_files
        self._bot_done_event = bot_done_event
        self._on_all_turns_done = on_all_turns_done
        self._inter_turn_pause_secs = inter_turn_pause_secs
        self._sample_rate = 0
        self._playback_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        await self.set_transport_ready(frame)
        # Start the playback loop
        self._playback_task = asyncio.create_task(self._playback_loop())

    async def cleanup(self):
        await super().cleanup()
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass

    async def _playback_loop(self):
        """Stream each audio file as a separate user turn."""
        try:
            for i, audio_file in enumerate(self._audio_files):
                turn_num = i + 1
                logger.info(f"[Turn {turn_num}/{len(self._audio_files)}] Playing: {audio_file}")

                pcm_data, file_sr, num_channels = read_wav_as_pcm16(
                    audio_file, self._sample_rate
                )

                # Signal user started speaking
                await self.push_frame(UserStartedSpeakingFrame())

                # Stream audio in chunks to simulate real-time
                chunk_samples = int(self._sample_rate * CHUNK_DURATION_MS / 1000)
                chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample
                offset = 0

                while offset < len(pcm_data):
                    chunk = pcm_data[offset : offset + chunk_bytes]
                    frame = InputAudioRawFrame(
                        audio=chunk,
                        sample_rate=file_sr,
                        num_channels=num_channels,
                    )
                    await self.push_audio_frame(frame)
                    offset += chunk_bytes
                    # Pace to ~real-time
                    await asyncio.sleep(CHUNK_DURATION_MS / 1000.0)

                # Signal user stopped speaking
                await self.push_frame(UserStoppedSpeakingFrame())

                logger.info(f"[Turn {turn_num}] Audio sent, waiting for bot response...")

                # Wait for bot to finish responding
                self._bot_done_event.clear()
                await self._bot_done_event.wait()

                # Pause between turns
                if i < len(self._audio_files) - 1:
                    await asyncio.sleep(self._inter_turn_pause_secs)

            logger.info("All turns complete.")
            self._on_all_turns_done.set()

        except asyncio.CancelledError:
            logger.info("Playback cancelled.")
        except Exception:
            logger.exception("Error in playback loop")
            self._on_all_turns_done.set()


class FileAudioOutputTransport(BaseOutputTransport):
    """Receives output audio from the pipeline. Optionally saves to file."""

    def __init__(
        self,
        params: TransportParams,
        bot_done_event: asyncio.Event,
        save_output: bool = False,
        output_dir: str = "output_audio",
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._bot_done_event = bot_done_event
        self._save_output = save_output
        self._output_dir = output_dir
        self._turn_count = 0
        self._current_audio_chunks: List[bytes] = []
        self._sample_rate = 0

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        # BotStoppedSpeakingFrame is generated internally by MediaSender and
        # pushed via push_frame (not process_frame), so we intercept it here.
        if isinstance(frame, BotStoppedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            logger.debug("Bot stopped speaking - signaling input transport")
            if self._save_output and self._current_audio_chunks:
                self._save_turn_audio()
            self._bot_done_event.set()
        await super().push_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Receive output audio. Optionally accumulate for saving."""
        if self._save_output:
            self._current_audio_chunks.append(frame.audio)
        return True

    def _save_turn_audio(self):
        """Save accumulated audio chunks to a WAV file."""
        import os

        os.makedirs(self._output_dir, exist_ok=True)
        self._turn_count += 1
        filepath = os.path.join(self._output_dir, f"bot_turn_{self._turn_count:03d}.wav")

        audio_data = b"".join(self._current_audio_chunks)
        self._current_audio_chunks.clear()

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_data)

        logger.info(f"Saved bot audio: {filepath}")

    async def cleanup(self):
        await super().cleanup()
        if self._save_output and self._current_audio_chunks:
            self._save_turn_audio()


class FileAudioTransport(BaseTransport):
    """Transport that reads user audio from WAV files for multi-turn eval.

    Args:
        audio_files: List of WAV file paths, one per user turn.
        params: Transport params (set audio_in_enabled=True, audio_out_enabled=True).
        save_bot_audio: If True, saves bot responses as WAV files.
        output_dir: Directory to save bot audio.
        inter_turn_pause_secs: Seconds to wait between turns.
    """

    def __init__(
        self,
        audio_files: List[str],
        params: TransportParams,
        save_bot_audio: bool = False,
        output_dir: str = "output_audio",
        inter_turn_pause_secs: float = 1.0,
    ):
        super().__init__()
        self._params = params
        self._audio_files = audio_files
        self._save_bot_audio = save_bot_audio
        self._output_dir = output_dir
        self._inter_turn_pause_secs = inter_turn_pause_secs

        # Shared events for coordinating input ↔ output
        self._bot_done_event = asyncio.Event()
        self._all_turns_done = asyncio.Event()

        self._input: Optional[FileAudioInputTransport] = None
        self._output: Optional[FileAudioOutputTransport] = None

        # Register event handlers so bot.py can use @transport.event_handler(...)
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = FileAudioInputTransport(
                audio_files=self._audio_files,
                params=self._params,
                bot_done_event=self._bot_done_event,
                on_all_turns_done=self._all_turns_done,
                inter_turn_pause_secs=self._inter_turn_pause_secs,
            )
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = FileAudioOutputTransport(
                params=self._params,
                bot_done_event=self._bot_done_event,
                save_output=self._save_bot_audio,
                output_dir=self._output_dir,
            )
        return self._output

    async def run(self):
        """Run the transport lifecycle: connect → play all turns → disconnect.

        Call this after setting up the pipeline and starting the runner.
        """
        # Simulate client connection
        await self._call_event_handler("on_client_connected", None)

        # Wait for all turns to complete
        await self._all_turns_done.wait()

        # Simulate client disconnection
        await self._call_event_handler("on_client_disconnected", None)
