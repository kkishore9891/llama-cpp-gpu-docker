# llama-cpp-python with GPU Acceleration in Docker

This repository provides a self-contained environment for running `llama-cpp-python` with NVIDIA GPU acceleration using Docker and Docker Compose. It's configured to handle Hugging Face authentication for accessing gated models like Gemma 3 QAT GGUF.

**Goal:** Run GGUF models via `llama-cpp-python` using the host's GPU, without cluttering the host system. Handles Hugging Face Hub authentication securely.

**Target Model Example:** `google/gemma-3-12b-it-qat-q4_0-gguf`

## Prerequisites (Host Machine)

1.  **Linux OS:** Tested on Ubuntu 24.04.
2.  **NVIDIA GPU:** A CUDA-compatible NVIDIA GPU.
3.  **NVIDIA Driver:** Latest recommended driver installed.
4.  **Docker Engine:** Installed and running.
5.  **NVIDIA Container Toolkit:** **Installed, configured, and verified working.** The command `docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi` should succeed on your host.
6.  **Hugging Face Hub Token:** A valid token from [HF Settings](https://huggingface.co/settings/tokens) with access permissions for the desired gated model.
7.  **VS Code & Dev Containers Extension:** (Optional, for Dev Container usage).

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone git@github.com:kkishore9891/llama-cpp-gpu-docker.git
    cd llama-cpp-gpu-docker
    ```
2.  **Create `.env` File:**
    * Copy the example file: `cp .env.example .env`
    * Edit the `.env` file (`nano .env` or your preferred editor).
    * Paste your Hugging Face Hub token into the file, replacing `your_hf_token_here`.
    ```dotenv
    # .env
    HUGGING_FACE_HUB_TOKEN=hf_YOUR_ACTUAL_TOKEN
    ```
    * The `.gitignore` file prevents your `.env` file (containing the secret token) from being committed to Git.

## Usage

You can run the environment using either Docker Compose directly or via the VS Code Dev Container feature. The container includes Python and `llama-cpp-python` compiled with CUDA support.

**Option 1: Using Docker Compose (Terminal)**

1.  **Build the Docker Image:**
    ```bash
    docker compose build
    ```
    * This builds the image defined in `.docker/llama-cpp/Dockerfile`. It compiles `llama-cpp-python` with CUDA support, which might take a few minutes.

2.  **Run the Python Script:**
    Use `docker compose run` to execute commands inside a new container based on the built image. It automatically reads the `.env` file.
    ```bash
    docker compose run --rm llama-cpp python app/run_gemma.py
    ```
    * `--rm`: Removes the container after the script finishes.
    * `llama-cpp`: The name of the service defined in `docker-compose.yml`.
    * `python app/run_gemma.py`: The command to run inside the container.
    * The first time you run this, `llama-cpp-python` will download the model (~7-8GB for 12B q4_0) using your HF token and cache it in the `./data/huggingface` directory. Subsequent runs will be faster.

3.  **Run an Interactive Shell (Optional):**
    ```bash
    docker compose run --rm llama-cpp bash
    ```
    This gives you a bash shell inside the container for interactive use or debugging.

**Option 2: Using VS Code Dev Containers**

1.  **Open the Folder in VS Code:**
    * Open the cloned `llama-cpp-gpu-docker` folder in VS Code.
2.  **Reopen in Container:**
    * VS Code should detect the `.devcontainer/devcontainer.json` file and prompt you: "Reopen folder to develop in a container." Click it.
    * (Alternative: Command Palette (`Ctrl+Shift+P`) > "Dev Containers: Reopen in Container").
3.  **Wait for Build/Start:** VS Code will build the image (if needed) and start the Dev Container.
4.  **Use the Integrated Terminal:**
    * Once running, open a terminal in VS Code (`Terminal > New Terminal`). This terminal is inside the container.
    * Verify GPU access: `nvidia-smi`
    * The `HUGGING_FACE_HUB_TOKEN` from your `.env` file is automatically available as an environment variable inside this terminal.
    * Run the script: `python app/run_gemma.py`
    * You can edit the `app/run_gemma.py` file in VS Code, and the changes are reflected inside the container immediately thanks to the volume mount.

## Notes

* **Model Caching:** Hugging Face models downloaded via `from_pretrained` are cached in `./data/huggingface` on your host (mounted to `/home/vscode/.cache/huggingface` inside).
* **Resource Requirements:** The Gemma 3 12B QAT model still requires significant RAM and VRAM, though less than larger models. Monitor your system resources.
* **`llama-cpp-python` Compilation:** The Dockerfile compiles `llama-cpp-python` with `CMAKE_ARGS="-DLLAMA_CUBLAS=on"`. If you need other backends (like OpenCL or Metal), adjust the `CMAKE_ARGS` in the Dockerfile and rebuild.
