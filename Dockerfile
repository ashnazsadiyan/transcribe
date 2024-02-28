FROM public.ecr.aws/lambda/python:3.11

COPY . .

# Create a directory for data
RUN mkdir /tmp/data


# Set the TRANSFORMERS_CACHE environment variable
ENV TRANSFORMERS_CACHE "/tmp/data"
ENV LIBROSA_CACHE_DIR "/tmp"
ENV NUMBA_CACHE_DIR "/tmp"

#COPY ffmpeg '/usr/share/'
#
#COPY Tool '/tmp'

RUN yum -y install git wget tar xz
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && tar xvf ffmpeg-release-amd64-static.tar.xz && mv ffmpeg-6.0-amd64-static/ffmpeg /usr/bin/ffmpeg && rm -Rf ffmpeg*
RUN pip install --no-cache-dir setuptools-rust
RUN pip install --no-cache-dir git+https://github.com/openai/whisper.git
RUN whisper --model_dir /usr/local --model medium audio >> /dev/null 2>&1; exit 0
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

#RUN python -m nltk.downloader punkt

# Copy app.py
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD for the Lambda function
CMD [ "app.handler" ]