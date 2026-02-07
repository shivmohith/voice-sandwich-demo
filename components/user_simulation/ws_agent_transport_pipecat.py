"""Pipecat WebSocket client transport wrapper for external agent simulations.

This wrapper uses Pipecat's native protobuf WebSocket protocol so a simulator
can connect directly to a Pipecat WebSocket server transport.
"""

import asyncio
from typing import Optional

from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.client import WebsocketClientParams, WebsocketClientTransport


class PipecatWSAgentTransport:
    def __init__(
        self,
        uri: str,
        *,
        additional_headers: Optional[dict[str, str]] = None,
    ):
        self._done_event = asyncio.Event()
        self._transport = WebsocketClientTransport(
            uri=uri,
            params=WebsocketClientParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                additional_headers=additional_headers,
                serializer=ProtobufFrameSerializer(),
            ),
        )

        @self._transport.event_handler("on_disconnected")
        async def _on_disconnected(_transport, _client):
            self._done_event.set()

    def input(self):
        return self._transport.input()

    def output(self):
        return self._transport.output()

    def event_handler(self, event_name: str):
        return self._transport.event_handler(event_name)

    async def run(self):
        await self._done_event.wait()

    async def stop(self):
        self._done_event.set()

    async def close(self):
        # Connection lifecycle is managed by the pipeline transport itself.
        return None
