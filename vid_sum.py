import os
import sys
from pytubefix import YouTube
import whisper

# venv/scripts/activate


ffmpeg_path = os.path.abspath("ffmpeg/bin")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]


yt = YouTube("https://www.youtube.com/watch?v=RjNTdLEECN4")
stream = yt.streams.filter(file_extension='mp4', progressive=True).first()
stream.download(filename="video.mp4")

model = whisper.load_model("tiny")
result = model.transcribe("video.mp4")
transcript = result ["text"]
