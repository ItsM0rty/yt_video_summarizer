import os
import sys
from pytubefix import YouTube
import whisper
import subprocess
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration
import cv2
from sentence_transformers import SentenceTransformer




# venv/scripts/activate


ffmpeg_path = os.path.abspath("ffmpeg/bin")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]


yt = YouTube("https://www.youtube.com/watch?v=GE-wXrciqqM")
stream = yt.streams.filter(file_extension='mp4', progressive=True).first()
videoFile = "video.mp4"
stream.download(filename=videoFile)

model = whisper.load_model("tiny")
result = model.transcribe(videoFile)
transcript = result ["text"]

output_dir = "frames"
interval = 2 

os.makedirs(output_dir, exist_ok=True)

command = [
    "ffmpeg", 
    "-i", 
    videoFile, 
    "-vf", 
    f"fps=1/{interval}", 
    f"{output_dir}/frame_%04d.jpg"
]

subprocess.run(command)


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

frames_dir = "frames"
frames = []

for fileName in sorted(os.listdir(frames_dir)):
    if fileName.endswith(".jpg"):
        path = os.path.join(frames_dir, fileName)
        frame = cv2.imread(path)
        frames.append(frame)



image_embeddings=[]
for frame in frames:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=image, return_tensors="pt")


    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    image_embeddings.append(outputs.squeeze(0))



text_model = SentenceTransformer("all-MiniLM-L6-v2")
text_embedding = text_model.encode(transcript, convert_to_tensor=True)


visual_vector = torch.mean(torch.stack(image_embeddings), dim=0)
combined_vector = torch.cat((text_embedding, visual_vector), dim=-1)

summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-base")
summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-base")

input_text = "summarize: " + transcript
input_ids = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length = 512, truncation=True)
summary_ids = summarizer_model.generate(input_ids, max_length=150)

summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)