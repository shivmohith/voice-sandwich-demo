# User Simulation for Pipecat WebSocket Agent

This mode connects the simulator to a Pipecat WebSocket server transport using
Pipecat's protobuf frame serializer.

## Run

```bash
cd components/user_simulation
uv sync
AGENT_WS_URL=ws://localhost:8765 uv run python bot_pipecat.py
```

Required env vars:

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`

Optional env vars:

- `MAX_TURNS` (default: `5`) - stops simulator after this many simulated-user turns
- `SIMULATOR_IDLE_TIMEOUT_SECS` (default: `20`) - stops simulator when no speech activity is observed
- `SIMULATION_AUDIO_DIR` (default: `simulation_audio`) - base directory for saved turn WAV files

Audio is saved by default under `simulation_audio/<timestamp>/`:

- `turn_XXX_simulated_user.wav`
- `turn_XXX_agent.wav`

`XXX` is the exchange index, starting from the simulated user's utterance and including the agent reply.
