# AGENTS.md

This is a voice agent repo. `./components/python` has the voice agent and `./components/web` is the frontend.

### Architecture
- Python backend is in `components/python/src/main.py` with an async pipeline:
  `STT -> Agent -> TTS`, implemented as `RunnableGenerator` stages.
- Events and types are in `components/python/src/events.py`.
- Frontend static build output is expected at `components/web/dist` and is served
  by the Python backend.

### Providers
- STT providers:
  - `assemblyai` (default) via `components/python/src/assemblyai_stt.py`
  - `deepgram` via `components/python/src/deepgram_stt.py`
  - Select with `STT_PROVIDER` env var.
- TTS providers:
  - `cartesia` (default) via `components/python/src/cartesia_tts.py`
  - `deepgram` via `components/python/src/deepgram_tts.py`
  - `elevenlabs` via `components/python/src/elevenlabs_tts.py`
  - Select with `TTS_PROVIDER` env var.

### Running Locally
- Python is typically run from `components/python`:
  - `uv run src/main.py`
- Web build required for serving the UI:
  - `make build-web` from repo root, or
  - `pnpm install && pnpm build` from `components/web`.

### Recent Changes (Feb 2026)
- Added Deepgram STT adapter (`deepgram_stt.py`) and provider switch in `main.py`.
- Added Deepgram TTS adapter (`deepgram_tts.py`) and provider switch in `main.py`.
- Adjusted import in `main.py` to local import for `cartesia_tts`.
- Added graceful `WebSocketDisconnect` handling in `main.py`.

### Voice Evals Harness (Feb 2026)
- Added a provider-agnostic, audio-only eval harness under `components/python/src/voice_evals/`:
  - `config.py`: typed config loader (`.toml` / `.json`) for run/audio/ws/simulator settings.
  - `audio_utils.py`: audio chunking, silence padding, and artifact writers (`.raw` + metadata JSON + WAV for PCM16).
  - `ws_agent.py`: single-websocket adapter for agent-under-test; sends raw audio and reads JSON events (`voice_agent_v1` default schema).
  - `simulator.py`: simulator abstraction with two modes:
    - `stt_model_tts` implemented (LLM text generation + TTS)
    - `realtime` adapter hook implemented (custom adapter via `module:Class`)
  - `run.py`: conversation runner, per-turn audio persistence, event JSONL logging, transcript output.
- Added sample config `components/python/voice_evals_config.toml`.
- Added README section for running evals.

### Voice Evals Behavior
- One websocket connection per conversation to the agent-under-test.
- User simulator controls conversation ending via token (`end_token`, default `END_CALL`).
- Turn boundary uses simulator audio + silence padding + pause (no fixed first-turn requirement).
- All run artifacts are written to `components/python/evals/results/<run_id>/`:
  - `config.json`
  - `debug.log`
  - `events/conversation_0001.jsonl`
  - `audio/conversation_0001/turn_XX_{user|assistant}.*`
  - `transcripts/conversation_0001.txt`

### Debugging / Stability Fixes Applied
- `voice_evals/run.py`:
  - Fixed context manager bug (`async with` + regular file context).
  - Added `.env` loading via `load_dotenv()`.
  - Added structured logging (`--log-level`, per-run `debug.log` + console logs).
- `voice_evals/simulator.py`:
  - Added TTS receive idle timeout so runs do not hang after provider sends final chunk/done.
  - Added Deepgram TTS option normalization (`pcm_s16le` -> `linear16`) and safer default sample rate for Deepgram TTS.
- `deepgram_tts.py`:
  - Made `container` optional (do not force `container=none`).
  - Added clearer websocket handshake error for invalid Deepgram TTS config.

### Important Config Note
- Simulator TTS provider (`[simulator.tts]` in `voice_evals_config.toml`) is independent from the app backend TTS provider (`TTS_PROVIDER` env var in `main.py` runtime).
