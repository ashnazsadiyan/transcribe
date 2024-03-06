import whisper
import ssl
import subprocess
from fastapi import FastAPI, HTTPException
from mangum import Mangum
import os
import tempfile
from pydub import AudioSegment

ssl._create_default_https_context = ssl._create_unverified_context

video_url = 'https://d8cele0fjkppb.cloudfront.net/ivs/v1/624618927537/h3DzFIBds0W6/2024/2/21/13/34/9426YpfiMxUD/media/hls/master.m3u8'


os.environ["TRANSFORMERS_CACHE"] = "/tmp/data"
os.environ['WHISPERS_CACHE_DIR'] = "/tmp"

app = FastAPI()
handler = Mangum(app)


@app.get('/download')
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
        with tempfile.NamedTemporaryFile(dir="/tmp",delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(stdout)

        if os.path.exists(temp_wav.name):
            # Perform your transcription or other operations here
            # model = whisper.load_model("base")
            # result = model.transcribe(temp_wav.name, fp16=False)
            # print("Answer:", result["text"])
            return {
                "transcribed": os.path.exists(temp_wav.name),
            }
        else:
            raise FileNotFoundError("Temporary WAV file not found")

    except Exception as e:
        print(e)
        # Handle exceptions and return an appropriate error response
        return HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.post('/transcribing')
def get_transcribing():
    try:
        command = [
            # '/usr/share/ffmpeg',
            'ffmpeg',
            '-i',
            video_url,
            '-b:a', '64k',
            '-f', 'wav',  # Force output format to WAV
            'pipe:1'  # Send output to stdout
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        with tempfile.NamedTemporaryFile(dir='/tmp', delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(stdout)

        model = whisper.load_model("base")
        result = model.transcribe(f"{temp_wav.name}", word_timestamps=True, fp16=False)
        print("Answer:", result["text"])
        return {
            "transcribed": result,
        }

    except Exception as e:
        print(e)
        # Handle exceptions and return an appropriate error response
        return HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.get('/index')
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
        with tempfile.NamedTemporaryFile(dir="/tmp",delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(stdout)

        model = whisper.load_model("base")
        result = model.transcribe(temp_wav.name, fp16=False)
        print("Answer:", result["text"])
        return {
            "transcribed": result["text"],
        }

    except Exception as e:
        print(e)
        # Handle exceptions and return an appropriate error response
        return HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
