"""Compatibility fixes for known Pipecat transport issues in this project."""

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessor


def apply_broadcast_frame_instance_patch() -> None:
    """Allow broadcast_frame() to accept a frame instance.

    Some Pipecat websocket transport paths call `broadcast_frame(frame_instance)`
    even though the API expects a frame class. This patch makes the method
    tolerant by routing frame instances to `broadcast_frame_instance()`.
    """

    if getattr(FrameProcessor, "_voice_sandwich_broadcast_patch_applied", False):
        return

    original = FrameProcessor.broadcast_frame

    async def broadcast_frame_compat(self, frame_cls, **kwargs):
        if isinstance(frame_cls, Frame):
            await self.broadcast_frame_instance(frame_cls)
            return
        await original(self, frame_cls, **kwargs)

    FrameProcessor.broadcast_frame = broadcast_frame_compat
    FrameProcessor._voice_sandwich_broadcast_patch_applied = True
