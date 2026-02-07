"""
Deepgram Real-Time Streaming STT Transform

Connects to Deepgram's WebSocket Streaming API for live speech-to-text.

Input: PCM 16-bit audio buffer (bytes)
Output: STT events (stt_chunk for partials, stt_output for final transcripts)
"""

import asyncio
import contextlib
import json
import os
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

import websockets
from websockets.client import WebSocketClientProtocol

from events import STTChunkEvent, STTEvent, STTOutputEvent


class DeepgramSTT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        encoding: str = "linear16",
        channels: int = 1,
        model: Optional[str] = None,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        endpointing: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("Deepgram API key is required")

        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.model = model or os.getenv("DEEPGRAM_MODEL") or "nova-3-general"
        self.interim_results = interim_results
        self.punctuate = punctuate
        self.smart_format = smart_format
        self.endpointing = endpointing

        self._ws: Optional[WebSocketClientProtocol] = None
        self._connection_signal = asyncio.Event()
        self._close_signal = asyncio.Event()

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        while not self._close_signal.is_set():
            _, pending = await asyncio.wait(
                [
                    asyncio.create_task(self._close_signal.wait()),
                    asyncio.create_task(self._connection_signal.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            with contextlib.suppress(asyncio.CancelledError):
                for task in pending:
                    task.cancel()

            if self._close_signal.is_set():
                break

            if self._ws and self._ws.close_code is None:
                self._connection_signal.clear()
                try:
                    async for raw_message in self._ws:
                        try:
                            message = json.loads(raw_message)
                        except json.JSONDecodeError as exc:
                            print(f"[DEBUG] DeepgramSTT JSON decode error: {exc}")
                            continue

                        message_type = message.get("type")

                        if message_type != "Results":
                            if message_type in {"Error", "Warning"}:
                                print(f"DeepgramSTT {message_type}: {message}")
                            continue

                        channel = message.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        transcript = ""
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "") or ""

                        if not transcript:
                            continue

                        if message.get("is_final"):
                            yield STTOutputEvent.create(transcript)
                        else:
                            yield STTChunkEvent.create(transcript)
                except websockets.exceptions.ConnectionClosed:
                    print("DeepgramSTT: WebSocket connection closed")

    async def send_audio(self, audio_chunk: bytes) -> None:
        ws = await self._ensure_connection()
        await ws.send(audio_chunk)

    async def close(self) -> None:
        if self._ws and self._ws.close_code is None:
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            await self._ws.close()
        self._ws = None
        self._close_signal.set()

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        if self._close_signal.is_set():
            raise RuntimeError(
                "DeepgramSTT tried establishing a connection after it was closed"
            )
        if self._ws and self._ws.close_code is None:
            return self._ws

        params = {
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "interim_results": str(self.interim_results).lower(),
            "punctuate": str(self.punctuate).lower(),
            "smart_format": str(self.smart_format).lower(),
        }
        params["model"] = self.model
        if self.endpointing is not None:
            params["endpointing"] = str(self.endpointing)

        url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
        self._ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Token {self.api_key}"}
        )

        self._connection_signal.set()
        return self._ws
