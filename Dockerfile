FROM python:3.9-slim

WORKDIR /app

# Install system dependencies first, before any other operations
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set memory and environment variables early
ENV PYTORCH_CPU_ONLY=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=256m
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model file using wget
RUN wget --no-verbose -O piano_transcription_inference_v1.pth https://huggingface.co/qiuqiangkong/piano_transcription/resolve/main/piano_transcription_inference_v1.pth

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with reduced worker count and memory limits
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "1", "--timeout-keep-alive", "30", "--memory-limit", "256"]
