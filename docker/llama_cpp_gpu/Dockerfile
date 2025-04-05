# Use an official NVIDIA CUDA development base image for Ubuntu 22.04 (Known Stable)
ARG CUDA_VERSION=12.4.1
# ARG OS_VERSION=24.04 # Comment out or remove old OS version
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Set frontend to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools, Python, pip, git, and other essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user 'vscode' for security and VS Code compatibility
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Optional: Add user to sudo group if needed inside container (requires installing sudo)
    && apt-get update && apt-get install -y sudo && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean

# Set up Python environment and install packages as the non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install huggingface_hub first for token management
RUN pip3 install --no-cache-dir --user huggingface_hub

# Install llama-cpp-python with CUDA support
# Ensure CMAKE_ARGS are set correctly for CUBLAS (GPU acceleration)
# Specify a version or remove ==... for latest
ARG LLAMA_CPP_PYTHON_VERSION="0.2.79" # Check for latest compatible version if needed
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1
# Use --break-system-packages if needed on newer pip/distro combos, or use a venv
# RUN python3 -m venv /home/$USERNAME/venv && . /home/$USERNAME/venv/bin/activate && \
RUN pip3 install --no-cache-dir --user llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION}

# Install other potentially useful Python libraries
RUN pip3 install --no-cache-dir --user requests Pillow

# Add local Python user bin to PATH
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Switch back to root only if necessary for final setup steps (e.g. apt install)
# USER root
# ...

# Set final working directory and default user for container execution
USER $USERNAME
WORKDIR /app

# Default command (useful for keeping container running with docker compose up -d)
# Can be overridden by docker compose run or devcontainer.json
CMD ["bash"]
