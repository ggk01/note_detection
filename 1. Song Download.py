from pytube import YouTube
from pydub import AudioSegment

# required ffmpeg -version

import yt_dlp

# Define options for yt-dlp
options = {
    'format': 'bestaudio/best',  # Best audio format available
    'outtmpl': 'audio.%(ext)s',  # Temporary file name and extension
    'postprocessors': [
        {
            'key': 'FFmpegExtractAudio',  # Extract audio using FFmpeg
            'preferredcodec': 'wav',     # Convert to WAV format
            'preferredquality': '192',   # Audio quality (optional)
        }
    ],
}

# URL of the YouTube video
url = "https://youtu.be/KFUW0p50P58?si=LUJ3Cr5R6jsYUoRP"

# Download the audio
with yt_dlp.YoutubeDL(options) as ydl:
    ydl.download([url])

print("Audio download complete!")
