import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import tempfile
from dotenv import load_dotenv
import torch
import subprocess
from tqdm import tqdm

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
            for i, (turn, _, speaker) in tqdm(enumerate(diarization.itertracks(yield_label=True)), desc="Transcribing audio", total=len(list(diarization.itertracks(yield_label=True)))):
                segment = full_audio[turn.start * 1000: turn.end * 1000]  # pydub works in ms
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    segment.export(tmp_wav.name, format="wav")
                    transcription = self.whisper_model.transcribe(tmp_wav.name)["text"].strip()
                    os.unlink(tmp_wav.name)  # Clean up

                out.write(f"Speaker {speaker}: {transcription}\n")

        print(f"Diarized transcription written to: {self.output_file_path}")

    def show_processing_animation(self, message, progress, total):
        import sys

        bar_length = 40
        filled_length = int(bar_length * progress / total)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r{message} [{bar}] {progress}/{total}')
        sys.stdout.flush()

    def run_pipeline(self, model_size="small"):
        total_steps = 4
        current_step = 1

        self.show_processing_animation("Converting audio to WAV...", current_step, total_steps)
        self.convert_to_wav()
        current_step += 1

        self.show_processing_animation("Loading models...", current_step, total_steps)
        self.load_models(model_size=model_size)
        current_step += 1

        self.show_processing_animation("Running diarization...", current_step, total_steps)
        diarization = self.run_diarization()
        current_step += 1

        self.show_processing_animation("Transcribing audio...", current_step, total_steps)
        self.transcribe(diarization)

# Example usage:
# transcription = SpeakerDiarizationTranscription("DE_Podcast.mp3", "Transcribe_output_file.txt", "your_huggingface_token")
# transcription.run_pipeline()
