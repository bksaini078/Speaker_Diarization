import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import tempfile
from dotenv import load_dotenv

# --- Config ---
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
AUDIO_FILE = "TheFutureMarkZuckerbergIsTryingToBuild.wav"  # mono 16kHz wav
OUTPUT_FILE = "TheFutureMarkZuckerbergIsTryingToBuild.txt"

# --- Load models ---
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HUGGINGFACE_TOKEN
)
whisper_model = whisper.load_model("small")  # Use "medium" or "large" for better accuracy

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
            transcription = whisper_model.transcribe(tmp_wav.name, fp16=False)["text"].strip()
            os.unlink(tmp_wav.name)  # Clean up

        out.write(f"{turn.start:.1f}s - {turn.end:.1f}s | Speaker {speaker}:\n{transcription}\n\n")

print(f"Diarized transcription written to: {OUTPUT_FILE}")
