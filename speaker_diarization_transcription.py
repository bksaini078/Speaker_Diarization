import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import tempfile
from dotenv import load_dotenv
import torch
import subprocess

class SpeakerTranscriptionDiarization:
    def __init__(self, input_audio_path, output_file_path, hf_token):
        self.input_audio_path = input_audio_path
        self.output_file_path = output_file_path
        self.hf_token = hf_token
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.diarization_pipeline = None
        self.whisper_model = None

    def convert_to_wav(self, output_path="podcast.wav"):
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Removed existing file: {output_path}")

        command = [
            "ffmpeg",
            "-i", self.input_audio_path,
            "-ac", "1",          # mono audio
            "-ar", "16000",      # 16kHz sample rate
            output_path
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Converted audio saved as: {output_path}")
            self.input_audio_path = output_path
        except subprocess.CalledProcessError as e:
            print("Error during audio conversion:", e)

    def load_models(self, model_size="small"):
        load_dotenv()
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=self.hf_token
        )
        self.diarization_pipeline.to(self.device)
        self.whisper_model = whisper.load_model(model_size, device="cpu")

    def run_diarization(self):
        return self.diarization_pipeline(self.input_audio_path)

    def transcribe(self, diarization):
        full_audio = AudioSegment.from_wav(self.input_audio_path)
        with open(self.output_file_path, "w") as out:
            for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
                segment = full_audio[turn.start * 1000: turn.end * 1000]  # pydub works in ms
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    segment.export(tmp_wav.name, format="wav")
                    transcription = self.whisper_model.transcribe(tmp_wav.name)["text"].strip()
                    os.unlink(tmp_wav.name)  # Clean up

                out.write(f"{turn.start:.1f}s - {turn.end:.1f}s | Speaker {speaker}:\n{transcription}\n\n")

        print(f"Diarized transcription written to: {self.output_file_path}")

    def run_pipeline(self, model_size="small"):
        self.convert_to_wav()
        self.load_models(model_size=model_size)
        diarization = self.run_diarization()
        self.transcribe(diarization)

# Example usage:
# transcription = SpeakerDiarizationTranscription("DE_Podcast.mp3", "Transcribe_output_file.txt", "your_huggingface_token")
# transcription.run_pipeline()
