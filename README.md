# Speaker Transcription Diarization

This project provides a Python class for speaker diarization and transcription of audio files. The `SpeakerTranscriptionDiarization` class encapsulates the entire process, including audio conversion, model loading, diarization, and transcription.

## Features

- Converts audio files to mono 16kHz WAV format using FFmpeg.
- Loads speaker diarization and Whisper models.
- Runs speaker diarization on the audio file.
- Transcribes each speaker's turn using the Whisper model.
- Writes the diarized transcription to an output file.

## Requirements

- Python 3.7+
- FFmpeg
- PyTorch
- Whisper
- Pyannote
- Pydub
- dotenv

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/speaker-transcription-diarization.git
   cd speaker-transcription-diarization
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure FFmpeg is installed on your system.

## Usage

1. Create a `.env` file in the project root directory and add your Hugging Face token:
   ```
   HF_TOKEN=your_huggingface_token
   ```

2. Use the `SpeakerTranscriptionDiarization` class in your script:

   ```python
   import os
   from src.speaker_diarization_transcription import SpeakerTranscriptionDiarization

   input_audio_path = "path/to/your/audio/file.mp3"
   output_file_path = "path/to/your/output/file.txt"
   hf_token = os.getenv("HF_TOKEN")

   transcription = SpeakerTranscriptionDiarization(input_audio_path, output_file_path, hf_token)
   transcription.run_pipeline(model_size="small")
   ```

## Example

Here is an example of how to use the `SpeakerTranscriptionDiarization` class:

```python
import os
from speaker_diarization_transcription import SpeakerTranscriptionDiarization

def main():
    input_audio_path = "TheFutureMarkZuckerbergIsTryingToBuild.mp3"
    output_file_path = "Transcribe_output_file.txt"
    hf_token = os.getenv("HF_TOKEN")

    transcription = SpeakerTranscriptionDiarization(input_audio_path, output_file_path, hf_token)
    transcription.run_pipeline(model_size="small")

if __name__ == "__main__":
    main()
```

