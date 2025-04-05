import os
from llama_cpp import Llama
# Optional: Explicit login if needed
# from huggingface_hub import login

# --- Hugging Face Authentication ---
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print("ERROR: HUGGING_FACE_HUB_TOKEN environment variable not set.")
    print("Please create a .env file in the project root with your token.")
    exit(1)
else:
    # Usually llama-cpp-python uses the env var directly for downloads.
    # Explicit login is generally not needed unless downloads fail.
    print("Hugging Face Hub token found via environment variable.")
    # Optional: Explicit Login (Uncomment if download fails)
    # try:
    #     login(token=hf_token)
    #     print("Hugging Face Hub login successful.")
    # except Exception as e:
    #     print(f"Warning: Error logging into Hugging Face Hub: {e}")


# --- Model Loading ---
# Ensure the repo_id is correct for the desired GGUF model
model_repo_id = "google/gemma-3-12b-it-qat-q4_0-gguf" # Your target model

print(f"\nLoading model: {model_repo_id}")
print("This may take a while on the first run as the model is downloaded...")
try:
    # Pass n_gpu_layers=-1 to offload all possible layers to GPU
    # Adjust n_ctx based on your needs and available VRAM
    llm = Llama.from_pretrained(
        repo_id=model_repo_id,
        # Llama.cpp usually auto-detects the filename in HF repos like this
        # filename="specific_filename.gguf", # Specify only if needed
        n_gpu_layers=-1,  # Offload all layers to GPU
        n_ctx=4096,       # Example context size, adjust as needed
        chat_format="gemma", # Specify chat format for Gemma models
        verbose=True      # Set to True for more detailed llama.cpp output
    )
    print("\nModel loaded successfully.")
except Exception as e:
    print(f"\nERROR loading model: {e}")
    print("Check if the model name is correct and you have access.")
    print("Ensure your HF token in the .env file is correct and has read permissions.")
    exit(1)

# --- Chat Completion (Text-Only Example) ---
# Note: The model specified seems text-only. Image input requires specific multimodal
# models and different handling in llama-cpp-python.
messages = [
     {"role": "system", "content": "You are a helpful AI assistant."},
     {"role": "user", "content": "Explain the difference between quantization and pruning in LLMs."}
 ]

print("\n--- Running Text Chat Completion ---")
try:
    response = llm.create_chat_completion(messages=messages)
    print("\n--- Response ---")
    # Check if response and choices are valid before accessing
    if response and 'choices' in response and len(response['choices']) > 0:
        message = response['choices'][0].get('message', {})
        content = message.get('content', 'No content found')
        print(content)
        print("\n--- Stats ---")
        usage = response.get('usage')
        if usage:
            print(f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"Total tokens: {usage.get('total_tokens', 'N/A')}")
    else:
        print("Invalid or empty response received.")
        print(f"Full response: {response}")

except Exception as e:
    print(f"\nERROR during chat completion: {e}")

print("\n--- Script Finished ---")
