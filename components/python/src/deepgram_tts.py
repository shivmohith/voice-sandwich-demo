"""
Deepgram Text-to-Speech Streaming

Python implementation of Deepgram's streaming TTS WebSocket API.
Converts text to PCM audio in real-time using WebSocket streaming.

Input: Text strings
Output: TTS events (tts_chunk for audio chunks)
"""

import asyncio
import contextlib
import json
import os
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

import websockets
from websockets.exceptions import InvalidStatus
from websockets.client import WebSocketClientProtocol

from events import TTSChunkEvent


class DeepgramTTS:
    _ws: Optional[WebSocketClientProtocol]
    _connection_signal: asyncio.Event
    _close_signal: asyncio.Event

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "aura-asteria-en",
        encoding: str = "linear16",
        container: Optional[str] = None,
        sample_rate: int = 24000,
        mip_opt_out: Optional[bool] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("Deepgram API key is required")

        self.model = os.getenv("DEEPGRAM_TTS_MODEL", model)
        self.encoding = os.getenv("DEEPGRAM_TTS_ENCODING", encoding)
        container_env = os.getenv("DEEPGRAM_TTS_CONTAINER")
        self.container = container_env if container_env is not None else container
        self.sample_rate = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", sample_rate))
        mip_env = os.getenv("DEEPGRAM_TTS_MIP_OPT_OUT")
        if mip_env is not None:
            self.mip_opt_out = mip_env.strip().lower() in {"1", "true", "yes"}
        else:
            self.mip_opt_out = mip_opt_out

        self._ws = None
        self._connection_signal = asyncio.Event()
        self._close_signal = asyncio.Event()

    async def send_text(self, text: Optional[str]) -> None:
        if text is None:
            return

        if not text.strip():
            return

        ws = await self._ensure_connection()
        await ws.send(json.dumps({"type": "Speak", "text": text}))
        await ws.send(json.dumps({"type": "Flush"}))

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
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
                        if isinstance(raw_message, (bytes, bytearray)):
                            if raw_message:
                                yield TTSChunkEvent.create(bytes(raw_message))
                            continue

                        try:
                            message = json.loads(raw_message)
                        except json.JSONDecodeError as exc:
                            print(f"[DEBUG] DeepgramTTS JSON decode error: {exc}")
                            continue

                        message_type = message.get("type")
                        if message_type in {"Warning", "Error"}:
                            print(f"[DEBUG] DeepgramTTS {message_type}: {message}")
                        # "Flushed" indicates end of current buffer, but the
                        # websocket can stay open for subsequent turns.
                except websockets.exceptions.ConnectionClosed:
                    print("DeepgramTTS: WebSocket connection closed")
                finally:
                    if self._ws and self._ws.close_code is None:
                        await self._ws.close()
                    self._ws = None

    async def close(self) -> None:
        if self._ws and self._ws.close_code is None:
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "Close"}))
            await self._ws.close()
        self._ws = None
        self._close_signal.set()

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        if self._close_signal.is_set():
            raise RuntimeError(
                "DeepgramTTS tried establishing a connection after it was closed"
            )
        if self._ws and self._ws.close_code is None:
            return self._ws

        params = {
            "model": self.model,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
        }
        if self.container:
            params["container"] = self.container
        if self.mip_opt_out is not None:
            params["mip_opt_out"] = str(self.mip_opt_out).lower()

        url = f"wss://api.deepgram.com/v1/speak?{urlencode(params)}"
        try:
            self._ws = await websockets.connect(
                url, additional_headers={"Authorization": f"Token {self.api_key}"}
            )
        except InvalidStatus as exc:
            raise RuntimeError(
                "Deepgram TTS websocket handshake failed. Check DEEPGRAM_API_KEY and "
                "DEEPGRAM_TTS_* settings (model/encoding/container/sample_rate)."
            ) from exc

        self._connection_signal.set()
        return self._ws
