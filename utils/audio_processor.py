import yt_dlp
from pydub import AudioSegment  #chunking
import os

DOWNLOAD_DIR="downloads"
os.makedirs(DOWNLOAD_DIR,exist_ok=True)

def download_youtube_audio(url:str)->str:
    output_path=os.path.join(DOWNLOAD_DIR,"%(title)s.%(ext)s")
    #youtube configs to download videos
    ydl_opts = {
        "format": "bestaudio/best",
        #where and how to save the downloaded file
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,  #view download progress bar, setting it to TRUE does not display it
    }

    #creating a YOUTUBE downloader object
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info=ydl.extract_info(url,download=True)
        filename=ydl.prepare_filename(info).replace(".webm",".wav").replace(".m4a",".wav")
    return filename

data=download_youtube_audio("https://youtu.be/T-D1OfcDW1M?si=EgubxcQyObKO7NvS")

def convert_to_wav(input_path:str)->str:
    """convert any audio/video file to WAV format using pydub"""
    output_path=os.path.splitext(input_path)[0]+"_converted.wav"
    audio=AudioSegment.from_file(input_path)
    audio=audio.set_channels(1).set_frame_rate(16000)  #16khz
    audio.export(output_path,format="wav")
    return output_path

print(convert_to_wav(data))