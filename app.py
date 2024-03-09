import whisper
import ssl
import subprocess
from fastapi import FastAPI, HTTPException
from mangum import Mangum
import io, os
import tempfile
from pydub import AudioSegment
import torch
from pydantic import BaseModel
from typing import List, Tuple

ssl._create_default_https_context = ssl._create_unverified_context

video_url = 'https://d8cele0fjkppb.cloudfront.net/ivs/v1/624618927537/h3DzFIBds0W6/2024/2/21/13/34/9426YpfiMxUD/media/hls/master.m3u8'

os.environ["TRANSFORMERS_CACHE"] = "/tmp/data"
os.environ['WHISPERS_CACHE_DIR'] = "/tmp"

os.makedirs("/tmp/data", exist_ok=True)

app = FastAPI()
handler = Mangum(app)


def extract_video_segment(input_video, time_stamp):
    start_time, end_time = time_stamp

    try:
        AudioSegment.converter = "ffmpeg"
        AudioSegment.ffmpeg = "ffmpeg"
        AudioSegment.ffprobe = "ffprobe"

        audio = AudioSegment.from_file(io.BytesIO(input_video))
        audio_segment = audio[start_time * 1000:end_time * 1000]
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_segment.export(temp_file.name, format="wav")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", download_root="/tmp").to(device)
        result = model.transcribe(temp_file.name, fp16=False)

        return {
            "transcribed": result["text"],
        }

    except Exception as e:
        print("Error:", str(e))
        return None


@app.get('/testing')
def get_text():
    try:
        command = [
            '/usr/bin/ffmpeg',
            # 'ffmpeg',
            '-i',
            video_url,
            '-b:a', '64k',
            '-f', 'wav',  # Force output format to WAV
            'pipe:1'  # Send output to stdout
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        with tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(stdout)

        time_stamps = [[0, 40]]
        extracted_videos = []

        for index, time_stamp in enumerate(time_stamps):
            extracted_video_path = extract_video_segment(stdout, time_stamp)
            if extracted_video_path:
                print(f"Video segment {index + 1} extracted successfully:", extracted_video_path)
                extracted_videos.append(extracted_video_path)
            else:
                print(f"Failed to extract video segment {index + 1}.")

        return {
            "transcribed": extracted_videos,
        }

    except Exception as e:
        print(e)
        # Handle exceptions and return an appropriate error response
        return HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


class Transcribing(BaseModel):
    times: List[Tuple[int, int]]
    url: str


@app.post('/transcribe')
def transcribe(transcribing: Transcribing):
    try:
        command = [
            # '/usr/bin/ffmpeg',
            'ffmpeg',
            '-i',
            transcribing.url,
            '-b:a', '64k',
            '-f', 'wav',  # Force output format to WAV
            'pipe:1'  # Send output to stdout
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        with tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(stdout)

        extracted_videos = []

        for index, time_stamp in enumerate(transcribing.times):
            extracted_video_path = extract_video_segment(stdout, time_stamp)
            if extracted_video_path:
                print(f"Video segment {index + 1} extracted successfully:", extracted_video_path)
                extracted_videos.append(extracted_video_path)
            else:
                print(f"Failed to extract video segment {index + 1}.")

        return {
            "transcribed": extracted_videos,
        }



    except Exception as e:
        print(e)
        # Handle exceptions and return an appropriate error response
        return HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
