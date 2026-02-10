#!/usr/bin/env python3
"""Simple VAPI WebSocket client - send/receive audio via VAPI's WebSocket transport."""

import argparse
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


def read_audio_pcm(path: str) -> bytes:
    """Read raw PCM int16 data from a WAV or raw .pcm file."""
    if path.endswith(".pcm"):
        return Path(path).read_bytes()
    with wave.open(path) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("Need mono 16-bit WAV")
        return w.readframes(w.getnframes())


async def run_client(audio_file: str | None = None, out_file: str = "received.wav"):
    call = create_call()
    ws_url = call["transport"]["websocketCallUrl"]

    async with websockets.connect(ws_url) as ws:
        async def send_audio():
            if audio_file:
                pcm = read_audio_pcm(audio_file)
                for i in range(0, len(pcm), 3200):
                    await ws.send(pcm[i : i + 3200])
            else:
                while True:
                    chunk = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.buffer.read, 3200
                    )
                    if not chunk:
                        break
                    await ws.send(chunk)

        async def recv_loop():
            """Receive binary audio or JSON control messages."""
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
                                wav = wave.open(out_file, "wb")
                                wav.setnchannels(1)
                                wav.setsampwidth(2)
                                wav.setframerate(16000)
                            elif obj.get("status") == "stopped":
                                recording = False
                                if wav is not None:
                                    wav.close()
                                    wav = None
                                    print(f"Saved to {out_file}", file=sys.stderr)
            finally:
                if wav is not None:
                    wav.close()

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(recv_loop())

        try:
            await asyncio.gather(send_task, recv_task)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await ws.send(json.dumps({"type": "end-call"}))
                print("Sent end-call.", file=sys.stderr)
            except websockets.ConnectionClosed:
                pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VAPI WebSocket audio client")
    p.add_argument("--file", "-f", help="WAV or raw PCM file to send (16kHz mono 16-bit)")
    p.add_argument("--out", "-o", default="received.wav", help="WAV file to save received audio")
    args = p.parse_args()
    try:
        asyncio.run(run_client(audio_file=args.file, out_file=args.out))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
