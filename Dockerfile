FROM python:3.10-slim

# Install system deps for faiss and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create index directory at runtime
RUN mkdir -p index

EXPOSE 7863

# HuggingFace Spaces expects the app on 0.0.0.0:7860 by default,
# but this service uses 7863 per spec.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7863"]
