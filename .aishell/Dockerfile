FROM aishell:foundation

# NVIDIA CUDA runtime (driver is mounted from host via nvidia-container-toolkit)
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-12-8 \
    libcublas-12-8 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswresample-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# --- Python + uv (uv-managed CPython 3.12) -------------------------

ENV UV_UNMANAGED_INSTALL=/usr/local/bin \
    UV_NO_MODIFY_PATH=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_BIN_DIR=/usr/local/bin

RUN set -eux; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    uv python install 3.12; \
    ln -sf "$(which python3.12)" /usr/local/bin/python; \
    ln -sf "$(which python3.12)" /usr/local/bin/python3
