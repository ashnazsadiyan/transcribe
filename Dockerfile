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

RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

#RUN python -m nltk.downloader punkt

# Copy app.py
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD for the Lambda function
CMD [ "app.handler" ]