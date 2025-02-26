FROM python:3.9-slim

WORKDIR /app

# Install system dependencies first, before any other operations
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set memory and environment variables early
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=256m
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory and download the model during build
RUN mkdir -p /app/models
RUN python -c "from transformers import AutoFeatureExtractor, AutoModelForAudioClassification; \
    model_name='modelo-base/piano-transcription-transformer'; \
    processor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir='/app/models'); \
    model = AutoModelForAudioClassification.from_pretrained(model_name, cache_dir='/app/models')"

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with reduced worker count and memory limits
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "1", "--timeout-keep-alive", "30", "--memory-limit", "256"]
