FROM --platform=$BUILDPLATFORM python:3.11-slim 

WORKDIR /artworks

# Install system dependencies required for some Python packages (like faiss-cpu)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    swig \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# Copy application files
COPY artwork_embeddings ./artwork_embeddings
COPY static ./static
COPY templates ./templates
COPY .env .
COPY app.py .

# Run the app
# ENTRYPOINT ["python3"]
# CMD ["app.py"]