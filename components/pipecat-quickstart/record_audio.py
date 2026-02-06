"""Record audio from microphone and save as WAV. Press Ctrl+C to stop."""

import wave
import pyaudio

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
OUTPUT_FILE = "audio_turns/turn_01.wav"


def record():
    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print(f"Recording... Press Ctrl+C to stop. Saving to {OUTPUT_FILE}")
    frames = []

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("\nStopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(OUTPUT_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        OUTPUT_FILE = sys.argv[1]
    record()
