FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for building torch-scatter/torch-sparse
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin torch to base image version to avoid ABI mismatch with scatter/sparse
COPY requirements.txt requirements_rl.txt ./
RUN pip install --no-cache-dir \
    "torch==2.1.0" \
    "numpy>=1.20.0,<2" \
    "tqdm>=4.60.0" \
    "gymnasium>=0.28.0" \
    "stable-baselines3>=2.0.0" \
    "matplotlib>=3.5.0" \
    "ortools>=9.9.0" \
    "gradio>=4.0.0"

# Install torch-geometric stack with prebuilt wheels matching torch 2.1.0 + cu121
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copy project
COPY . .

CMD ["bash"]
