#!/usr/bin/env python3
"""VAPI WebSocket client - send multiple audio files as sequential conversation turns."""

import asyncio
import json
import os
import sys
import wave
from pathlib import Path

import httpx
import websockets

# Add parent for dotenv
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.environ["VAPI_API_KEY"]
ASSISTANT_ID = "02bacaad-1a2f-46b9-9086-973b5c608054"
API_BASE = "https://api.vapi.ai"

AUDIO_TURNS = [
    "../audio_turns/input_math_1_e11labs.wav",
    "../audio_turns/input_math_2_e11labs.wav",
]
OUT_DIR = "responses"


def create_call() -> dict:
    """Create WebSocket call via VAPI API. Returns call object with websocketCallUrl."""
    resp = httpx.post(
        f"{API_BASE}/call",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "assistantId": ASSISTANT_ID,
            "transport": {
                "provider": "vapi.websocket",
                "audioFormat": {
                    "format": "pcm_s16le",
                    "container": "raw",
                    "sampleRate": 16000,
                },
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def read_wav_pcm(path: str) -> bytes:
    """Read raw PCM from WAV (16-bit mono). Resample not supported."""
    with wave.open(path) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("Need mono 16-bit WAV")
        return w.readframes(w.getnframes())


async def run_client(audio_files: list[str], out_dir: str = "."):
    call = create_call()
    ws_url = call["transport"]["websocketCallUrl"]

    async with websockets.connect(ws_url) as ws:
        try:
            for i, audio_file in enumerate(audio_files):
                print(f"\n--- Turn {i + 1}: sending {audio_file} ---", file=sys.stderr)

                # Send user audio
                pcm = read_wav_pcm(audio_file)
                for offset in range(0, len(pcm), 3200):
                    await ws.send(pcm[offset : offset + 3200])

                # Wait for assistant response using speech-update control messages
                out_path = os.path.join(out_dir, f"response_{i + 1}.wav")
                recording = False
                wav = None
                try:
                    async for msg in ws:
                        if isinstance(msg, bytes):
                            if recording and wav is not None:
                                wav.writeframes(msg)
                        else:
                            obj = json.loads(msg)
                            print(f"[control] {obj}", file=sys.stderr)
                            if obj.get("type") == "speech-update" and obj.get("role") == "assistant":
                                if obj.get("status") == "started":
                                    recording = True
                                    wav = wave.open(out_path, "wb")
                                    wav.setnchannels(1)
                                    wav.setsampwidth(2)
                                    wav.setframerate(16000)
                                elif obj.get("status") == "stopped":
                                    recording = False
                                    if wav is not None:
                                        wav.close()
                                        wav = None
                                    print(f"Saved to {out_path}", file=sys.stderr)
                                    break  # turn done, move to next
                finally:
                    if wav is not None:
                        wav.close()
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await ws.send(json.dumps({"type": "end-call"}))
                print("Sent end-call.", file=sys.stderr)
            except websockets.ConnectionClosed:
                pass


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        asyncio.run(run_client(audio_files=AUDIO_TURNS, out_dir=OUT_DIR))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
