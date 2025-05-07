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
    """
    A class to perform speaker diarization and transcription on audio files.

    Attributes:
        input_audio_path (str): Path to the input audio file.
        output_file_path (str): Path to the output transcription file.
        hf_token (str): Hugging Face token for accessing models.
        device (torch.device): Device to run the models on (mps, cuda, or cpu).
        diarization_pipeline (Pipeline): Speaker diarization pipeline.
        whisper_model (whisper.Model): Whisper model for transcription.
    """

    def __init__(self, input_audio_path, output_file_path, hf_token):
        """
        Initializes the SpeakerTranscriptionDiarization class.

        Args:
            input_audio_path (str): Path to the input audio file.
            output_file_path (str): Path to the output transcription file.
            hf_token (str): Hugging Face token for accessing models.
        """
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
        """
        Converts the input audio file to mono 16kHz WAV format.

        Args:
            output_path (str): Path to save the converted audio file.
        """
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
        """
        Loads the speaker diarization and Whisper models.

        Args:
            model_size (str): Size of the Whisper model ("small", "medium", "large").
        """
        load_dotenv()
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=self.hf_token
        )
        self.diarization_pipeline.to(self.device)
        self.whisper_model = whisper.load_model(model_size, device="cpu") # need to change here while running in linux system

    def run_diarization(self):
        """
        Runs speaker diarization on the input audio file.

        Returns:
            diarization: The diarization result.
        """
        return self.diarization_pipeline(self.input_audio_path)

    def transcribe(self, diarization):
        """
        Transcribes each speaker's turn using the Whisper model.

        Args:
            diarization: The diarization result.
        """
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
        """
        Displays a processing animation in the console.

        Args:
            message (str): The message to display.
            progress (int): The current progress.
            total (int): The total progress.
        """
        import sys

        bar_length = 40
        filled_length = int(bar_length * progress / total)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r{message} [{bar}] {progress}/{total}')
        sys.stdout.flush()

    def run_pipeline(self, model_size="small"):
        """
        Runs the complete pipeline for speaker diarization and transcription.

        Args:
            model_size (str): Size of the Whisper model ("small", "medium", "large").
        """
        total_steps = 4
        current_step = 1

        try:
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
        except Exception as e:
            print(f"An error occurred: {e}")

    def run_batch_pipeline(self, input_folder, output_folder, model_size="small"):
        """
        Runs the batch processing pipeline for speaker diarization and transcription.

        Args:
            input_folder (str): Path to the folder containing input audio files.
            output_folder (str): Path to the folder to save output transcription files.
            model_size (str): Size of the Whisper model ("small", "medium", "large").
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        audio_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp3', '.wav', '.flac'))]

        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            input_audio_path = os.path.join(input_folder, audio_file)
            output_file_path = os.path.join(output_folder, os.path.splitext(audio_file)[0] + ".txt")

            print(f"Processing {audio_file}...")
            self.input_audio_path = input_audio_path
            self.output_file_path = output_file_path
            self.run_pipeline(model_size=model_size)


