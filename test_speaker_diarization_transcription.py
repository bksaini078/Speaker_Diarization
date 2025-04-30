import os
from speaker_diarization_transcription import SpeakerTranscriptionDiarization

def test_speaker_diarization_transcription():
    input_audio_path = "TheFutureMarkZuckerbergIsTryingToBuild.mp3"
    output_file_path = "test_transcription_output.txt"
    hf_token = os.getenv("HF_TOKEN")

    transcription = SpeakerTranscriptionDiarization(input_audio_path, output_file_path, hf_token)
    transcription.run_pipeline(model_size="small")

    assert os.path.exists(output_file_path), "Output file was not created"
    with open(output_file_path, "r") as f:
        content = f.read()
        assert "Speaker" in content, "Transcription does not contain speaker information"

    print("Test passed: Transcription output file created successfully with speaker information.")

if __name__ == "__main__":
    test_speaker_diarization_transcription()
