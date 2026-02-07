import base64
import time
from dataclasses import asdict, dataclass
from typing import Literal, Union


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class UserInputEvent:
    type: Literal["user_input"]
    audio: bytes
    ts: int

    @classmethod
    def create(cls, audio: bytes) -> "UserInputEvent":
        return cls(type="user_input", audio=audio, ts=_now_ms())


@dataclass
class STTChunkEvent:
    type: Literal["stt_chunk"]
    transcript: str
    ts: int

    @classmethod
    def create(cls, transcript: str) -> "STTChunkEvent":
        return cls(type="stt_chunk", transcript=transcript, ts=_now_ms())


@dataclass
class STTOutputEvent:
    type: Literal["stt_output"]
    transcript: str
    ts: int

    @classmethod
    def create(cls, transcript: str) -> "STTOutputEvent":
        return cls(type="stt_output", transcript=transcript, ts=_now_ms())


STTEvent = Union[STTChunkEvent, STTOutputEvent]


@dataclass
class AgentChunkEvent:
    type: Literal["agent_chunk"]
    text: str
    ts: int

    @classmethod
    def create(cls, text: str) -> "AgentChunkEvent":
        return cls(type="agent_chunk", text=text, ts=_now_ms())


@dataclass
class AgentEndEvent:
    type: Literal["agent_end"]
    ts: int

    @classmethod
    def create(cls) -> "AgentEndEvent":
        return cls(type="agent_end", ts=_now_ms())


@dataclass
class ToolCallEvent:
    type: Literal["tool_call"]
    id: str
    name: str
    args: dict
    ts: int

    @classmethod
    def create(cls, id: str, name: str, args: dict) -> "ToolCallEvent":
        return cls(type="tool_call", id=id, name=name, args=args, ts=_now_ms())


@dataclass
class ToolResultEvent:
    type: Literal["tool_result"]
    tool_call_id: str
    name: str
    result: str
    ts: int

    @classmethod
    def create(cls, tool_call_id: str, name: str, result: str) -> "ToolResultEvent":
        return cls(
            type="tool_result",
            tool_call_id=tool_call_id,
            name=name,
            result=result,
            ts=_now_ms(),
        )


AgentEvent = Union[AgentChunkEvent, AgentEndEvent, ToolCallEvent, ToolResultEvent]


@dataclass
class TTSChunkEvent:
    type: Literal["tts_chunk"]
    audio: bytes
    ts: int

    @classmethod
    def create(cls, audio: bytes) -> "TTSChunkEvent":
        return cls(type="tts_chunk", audio=audio, ts=_now_ms())


VoiceAgentEvent = Union[UserInputEvent, STTEvent, AgentEvent, TTSChunkEvent]


def event_to_dict(event: VoiceAgentEvent) -> dict:
    d = asdict(event)
    if isinstance(event, UserInputEvent):
        d.pop("audio", None)
    elif isinstance(event, TTSChunkEvent):
        d["audio"] = base64.b64encode(event.audio).decode("ascii")
    if isinstance(event, ToolResultEvent):
        d["toolCallId"] = d.pop("tool_call_id")
    return d
