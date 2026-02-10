from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play, save
import os

load_dotenv(override=True)

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVEN_API_KEY"),
)

audio = elevenlabs.text_to_speech.convert(
    text="Hello, if you add 1 + 1?",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="wav_16000",
)

save(audio, "input_math_1_e11labs.wav")
# play(audio)

