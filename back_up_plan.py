# import whisper

# model = whisper.load_model("turbo")
# result = model.transcribe("TheFutureMarkZuckerbergIsTryingToBuild.mp3")
# print(result["text"])


from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "cam++"

# model = AutoModel(
#     model=model_dir,
#     vad_model="fsmn-vad",
#     vad_kwargs={"max_single_segment_time": 30000},
#     device="cuda:0",
# )
model = AutoModel(model='iic/Whisper-large-v3', model_revision="v2.0.5",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  spk_model="cam++", spk_model_revision="v2.0.2"
                  )
res = model.generate(input="TheFutureMarkZuckerbergIsTryingToBuild.mp3", 
                     batch_size_s=300)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
transcript_data = res[0]
# print(f"Len of trancript data {len(transcript_data)}")
# print(res)
merged_transcript = []
current_text = ""
speaker=None
for i, segment in enumerate(res):
    # print(f"{i},{segment}")
    # speaker = f"Speaker {segment['spk']}"
    text = segment['text']
    sub_segment= segment["sentence_info"]
    current_text= ""
    print(f"{i}")
    for item in sub_segment:
        # print(item)
        if speaker is None:
            # First segment
            current_text = item["text"]
            speaker=item['spk']
        elif speaker == item['spk']:
            # Same speaker, append text
            current_text += item["text"]

        else:
            # Speaker changed, add previous speaker's text and start new speaker
            print("Speaker Changed")
            # print(f"Speaker {speaker} : {current_text}")
            merged_transcript.append(f"Speaker_{speaker}: {current_text}")
            current_text= item["text"]
            speaker = item['spk']

    # If it's the last segment, add the current speaker's text
    if i == len(transcript_data) - 1:
        merged_transcript.append(f"{speaker}: {current_text}")

formatted_transcript = "\n".join(merged_transcript)
print(f"formatted_transcript: {formatted_transcript}")
# print(current_text)
