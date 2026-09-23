import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from google.colab import drive
from huggingface_hub import login
from google.colab import userdata
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"

drive.mount("/content/drive")
audio_filename = "/content/drive/MyDrive/<name-of-your-audio-file>.mp3"

hf_token = userdata.get('HuggingFace')
login(hf_token, add_to_git_credential=True)

audio_file = open(audio_filename, "rb")

from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    dtype=torch.float16,
    device='cuda',
    return_timestamps=True
)

result = pipe(audio_filename)
transcription = result["text"]
print(transcription)

open_source_transcription = transcription

from groq import Groq

groq_api_key = userdata.get('GROQ_API_KEY')

groq_client = Groq(api_key=groq_api_key)

transcription = groq_client.audio.transcriptions.create(
    file=("denver_extract.mp3", audio_file),
    model="whisper-large-v3",
    response_format="text"
)

print(transcription)

display(Markdown(open_source_transcription))
print("\n\n")
display(Markdown(transcription))

system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""

user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription:
{transcription}
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

response = tokenizer.decode(outputs[0])

display(Markdown(response)) 