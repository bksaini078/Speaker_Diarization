import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import tempfile
from dotenv import load_dotenv
import torch

import subprocess

def convert_to_wav(input_path, output_path="podcast.wav"):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",          # mono audio
        "-ar", "16000",      # 16kHz sample rate
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Converted audio saved as: {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error during audio conversion:", e)

# Example usage:
convert_to_wav("DE_Podcast.mp3")


# --- Config ---
load_dotenv()


#setting the device 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
AUDIO_FILE = "podcast.wav"#"TheFutureMarkZuckerbergIsTryingToBuild.wav"  # mono 16kHz wav
OUTPUT_FILE = "Transcribe_output_file.txt"

# --- Load models ---
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HUGGINGFACE_TOKEN
)
diarization_pipeline.to(device)
whisper_model = whisper.load_model("small",device="cpu")  # Use "medium" or "large" for better accuracy

# --- Run diarization ---
diarization = diarization_pipeline(AUDIO_FILE)

# --- Load audio ---
full_audio = AudioSegment.from_wav(AUDIO_FILE)

# --- Transcribe per speaker turn ---
with open(OUTPUT_FILE, "w") as out:
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        segment = full_audio[turn.start * 1000: turn.end * 1000]  # pydub works in ms
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            segment.export(tmp_wav.name, format="wav")
            transcription = whisper_model.transcribe(tmp_wav.name)["text"].strip()
            os.unlink(tmp_wav.name)  # Clean up

        out.write(f"{turn.start:.1f}s - {turn.end:.1f}s | Speaker {speaker}:\n{transcription}\n\n")

print(f"Diarized transcription written to: {OUTPUT_FILE}")
