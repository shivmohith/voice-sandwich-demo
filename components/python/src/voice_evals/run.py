from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

from voice_evals.audio_utils import silence_bytes, write_audio_artifact
from voice_evals.config import EvalConfig, load_config
from voice_evals.simulator import create_simulator, transcribe_audio
from voice_evals.ws_agent import AgentWebSocketClient

logger = logging.getLogger(__name__)


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _configure_logging(run_dir: Path, level: str) -> Path:
    log_path = run_dir / "debug.log"
    logging.root.handlers.clear()
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


async def run_eval(config: EvalConfig, *, log_level: str = "INFO") -> Path:
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = config.run.name or run_timestamp
    run_dir = config.run.results_dir / run_name
    events_dir = run_dir / "events"
    audio_dir = run_dir / "audio"
    transcripts_dir = run_dir / "transcripts"

    events_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_logging(run_dir, log_level)
    logger.info("Run directory: %s", run_dir)
    logger.info("Debug log: %s", log_path)
    logger.info("Agent websocket URL: %s", config.agent_ws.url)
    logger.info("Simulator mode: %s", config.simulator.mode)

    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(_serialize(config), indent=2), encoding="utf-8"
    )

    conversation_id = "conversation_0001"
    transcript_path = transcripts_dir / f"{conversation_id}.txt"
    event_log_path = events_dir / f"{conversation_id}.jsonl"

    simulator = create_simulator(
        simulator_config=config.simulator,
        audio_config=config.audio,
        end_token=config.run.end_token,
    )
    logger.info("Simulator initialized")

    transcript_lines: list[str] = []
    last_assistant_text: str | None = None
    last_assistant_audio: bytes = b""

    async with AgentWebSocketClient(
        url=config.agent_ws.url,
        headers=config.agent_ws.headers,
        schema=config.agent_ws.schema,
        input_mode=config.agent_ws.input_mode,
        output_mode=config.agent_ws.output_mode,
        tts_chunk_field=config.agent_ws.tts_chunk_field,
    ) as agent_ws:
        logger.info("Connected to agent websocket")
        with event_log_path.open("w", encoding="utf-8") as event_log:
            for turn_index in range(1, config.run.max_turns + 1):
                logger.info("Turn %s started", turn_index)
                assistant_text_for_sim = last_assistant_text
                if (
                    turn_index > 1
                    and not assistant_text_for_sim
                    and last_assistant_audio
                    and config.audio.output_format == "pcm16"
                ):
                    logger.info(
                        "Turn %s: assistant text missing, running STT on prior assistant audio (%s bytes)",
                        turn_index,
                        len(last_assistant_audio),
                    )
                    assistant_text_for_sim = await transcribe_audio(
                        audio_bytes=last_assistant_audio,
                        provider=config.simulator.stt.provider,
                        sample_rate_hz=config.audio.output_sample_rate_hz,
                        audio_format=config.audio.output_format,
                        chunk_ms=config.audio.chunk_ms,
                    )
                    logger.info(
                        "Turn %s: STT transcript='%s'",
                        turn_index,
                        assistant_text_for_sim,
                    )

                logger.info("Turn %s: generating simulator user turn", turn_index)
                user_turn = await simulator.next_turn(
                    assistant_text_for_sim, turn_index
                )
                logger.info(
                    "Turn %s: simulator text='%s' audio_bytes=%s should_end=%s",
                    turn_index,
                    user_turn.text,
                    len(user_turn.audio_bytes),
                    user_turn.should_end,
                )
                transcript_lines.append(f"TURN {turn_index} USER: {user_turn.text}")

                turn_dir = audio_dir / conversation_id
                write_audio_artifact(
                    turn_dir,
                    f"turn_{turn_index:02d}_user",
                    user_turn.audio_bytes,
                    config.audio.input_format,
                    config.audio.input_sample_rate_hz,
                )

                padding = silence_bytes(
                    config.audio.input_format,
                    config.audio.silence_padding_ms,
                    config.audio.input_sample_rate_hz,
                )
                await agent_ws.send_audio(
                    user_turn.audio_bytes + padding,
                    chunk_ms=config.audio.chunk_ms,
                    sample_rate_hz=config.audio.input_sample_rate_hz,
                    audio_format=config.audio.input_format,
                    real_time=config.audio.real_time,
                )
                logger.info(
                    "Turn %s: sent user audio + padding (%s + %s bytes)",
                    turn_index,
                    len(user_turn.audio_bytes),
                    len(padding),
                )
                if config.audio.pause_after_turn_ms > 0:
                    await asyncio.sleep(config.audio.pause_after_turn_ms / 1000.0)
                    logger.info(
                        "Turn %s: pause_after_turn_ms=%s completed",
                        turn_index,
                        config.audio.pause_after_turn_ms,
                    )

                assistant_result = await agent_ws.collect_turn(
                    idle_timeout_ms=config.audio.assistant_idle_timeout_ms,
                    log_file=event_log,
                    turn_index=turn_index,
                )
                logger.info(
                    "Turn %s: assistant response text_len=%s audio_bytes=%s",
                    turn_index,
                    len(assistant_result.assistant_text),
                    len(assistant_result.assistant_audio_bytes),
                )
                last_assistant_text = assistant_result.assistant_text
                last_assistant_audio = assistant_result.assistant_audio_bytes

                transcript_lines.append(
                    f"TURN {turn_index} ASSISTANT: {assistant_result.assistant_text}"
                )
                if assistant_result.assistant_audio_bytes:
                    write_audio_artifact(
                        turn_dir,
                        f"turn_{turn_index:02d}_assistant",
                        assistant_result.assistant_audio_bytes,
                        config.audio.output_format,
                        config.audio.output_sample_rate_hz,
                    )

                if user_turn.should_end:
                    logger.info("Turn %s: simulator requested end token, stopping", turn_index)
                    break

    transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
    logger.info("Transcript written: %s", transcript_path)
    logger.info("Event log written: %s", event_log_path)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run voice eval harness.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_name:
        config = EvalConfig(
            run=config.run.__class__(
                name=args.run_name,
                results_dir=config.run.results_dir,
                max_turns=config.run.max_turns,
                end_token=config.run.end_token,
            ),
            audio=config.audio,
            agent_ws=config.agent_ws,
            simulator=config.simulator,
        )
    run_dir = asyncio.run(run_eval(config, log_level=args.log_level))
    print(f"Wrote results to {run_dir}")


if __name__ == "__main__":
    main()
