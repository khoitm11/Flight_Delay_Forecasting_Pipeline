FROM python:3.12-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app_flight_delay

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./saved_models ./saved_models

CMD ["python", "app/app.py", \
     "--airport", "JFK", \
     "--arr_cancelled", "0", \
     "--weather_ct_rate", "0.01" \
    ]