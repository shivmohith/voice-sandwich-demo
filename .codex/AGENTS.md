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
