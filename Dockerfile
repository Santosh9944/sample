FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by opencv-headless
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create uploads folder
RUN mkdir -p static/uploads

# HuggingFace Spaces uses port 7860
EXPOSE 7860

ENV PORT=7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "--preload"]
