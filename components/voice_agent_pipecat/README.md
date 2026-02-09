# Voice Agent (Pipecat)

Standalone Pipecat-based voice agent implementation.

## Run

```bash
cd components/voice_agent_pipecat
uv sync
uv run src/main.py
```

The server starts a Pipecat WebSocket transport at `ws://0.0.0.0:8765` by default.

## Environment

Required:

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`

Optional:

- `PIPECAT_AGENT_HOST` (default: `0.0.0.0`)
- `PIPECAT_AGENT_PORT` (default: `8765`)
- `PIPECAT_AGENT_LLM_MODEL` (default: `gpt-4o-mini`)
- `PIPECAT_AGENT_INPUT_SAMPLE_RATE` (default: `16000`)
- `PIPECAT_AGENT_TTS_SAMPLE_RATE` (default: `16000`)
- `DEEPGRAM_TTS_MODEL` (default: `aura-asteria-en`)
- `DEEPGRAM_STT_MODEL` (default: `nova-3-general`)
- `DEEPGRAM_STT_ENDPOINTING_MS` (default: `500`)
- `DEEPGRAM_STT_UTTERANCE_END_MS` (default: `1000`)
