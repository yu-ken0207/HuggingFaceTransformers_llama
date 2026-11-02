# LLaMA API Service with FastAPI

**English** | [繁體中文](README.zh-TW.md)

[![Checks](https://github.com/yu-ken0207/HuggingFaceTransformers_llama/actions/workflows/check.yml/badge.svg)](https://github.com/yu-ken0207/HuggingFaceTransformers_llama/actions/workflows/check.yml)

Serves Meta Llama 3.1 8B Instruct as a FastAPI service for robot task planning: `/generate` for chat-style generation and `/generate_with_function` for tool calling with a JSON schema (for example, breaking "put the book on the side table" into steps). A second service loads two LoRA adapters fine-tuned for task decomposition and sub-task planning; the adapters are included in this repo and were trained with [LlamaLoRAFinetuner](https://github.com/yu-ken0207/LlamaLoRAFinetuner).

**Built with Llama**: the LoRA weights in the two `Llama-3.1-8B-Instruct-bnb-4bit_*` folders are fine-tuned from Meta Llama 3.1. See the license section at the bottom.

The service has two endpoints:

- Simple text generation (`/generate`)
- Generation with tools (`/generate_with_function`), where tools are defined per request

## Features

### Text generation
- `/generate` returns a reply to the given messages.

### Tools defined per request
- Define tools with a JSON schema to handle tasks such as breaking a task into steps (`/generate_with_function`).

### Memory options for large models
- Supports `torch.float16` and 8-bit quantization to reduce memory use.

## Requirements

- Python 3.11+ (tested with 3.12)
- A CUDA-capable GPU
- Install torch for your GPU's CUDA version first (commands for other versions are on the [PyTorch website](https://pytorch.org/get-started/locally/)), then the other packages. `environment.yml` only creates the Python environment:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # CUDA 13.0
pip install -r requirements.txt
```

`requirements.txt` lists versions tested in November 2026 (transformers 5.17.0, unsloth 2026.10.11, and others). Both services' `/generate` and `/generate_with_function` were tested on Linux ARM64 with an NVIDIA Blackwell GPU and CUDA 13.0 (the base-model service was tested with Llama 3.2 1B Instruct instead of 8B, in both float32 and 8-bit). On x86, follow the same steps with the torch install command for your CUDA version; this was not tested on real hardware, but GitHub Actions checks that the dependency versions resolve on x86 with Python 3.11 and 3.12.

> **Note**: unsloth currently supports only datasets versions before 5.0. The datasets 4.8.5 and fsspec versions it brings in each have one known vulnerability (in folder-based dataset builders and in Kerchunk reference files), which this service doesn't use. The service also has no authentication: run it only on your own computer or a trusted network, never directly on the internet.

## Setup

### 1. Environment
To avoid memory fragmentation and make better use of the GPU, set:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 2. Start the API

Base Llama 3.1 (with tool calling). First set `model_id` in `llama_api_service_function.py` to the path of the model:

```bash
uvicorn llama_api_service_function:app --host 0.0.0.0 --port 8000
```

Or load the two LoRA adapters included in this repo (task decomposition and sub-task planning; needs `unsloth`):

```bash
uvicorn llama_api_service_fintune:app --host 0.0.0.0 --port 8000
```

## API endpoints

### 1. `/generate`
- **Description**: generates a reply to the user's messages.
- **Request example**:

  ```json
  {
    "messages": [
      {"role": "system", "content": "Role: Professional Task Planning Robot"},
      {"role": "user", "content": "Move a knife and cup to a counter."}
    ],
    "max_new_tokens": 256
  }
  ```
- **Response example**:

  ```json
  {
    "role": "assistant",
    "content": "1. Find the knife. 2. Pick up the knife. 3. Find the cup. 4. Pick up the cup. 5. Walk to the counter. 6. Place the knife and cup on the counter."
  }
  ```

### 2. `/generate_with_function`
- **Description**: lets the caller define tools for more complex generation tasks.
- **Request example**:

  ```json
  {
    "messages": [
      {"role": "system", "content": "You will receive a task. Decompose it into multiple steps."},
      {"role": "user", "content": "Put the book on the side table."}
    ],
    "tools": [
      {
        "name": "decompose_task",
        "description": "Decompose the task into subtasks and return a JSON object containing the list of subtasks.",
        "parameters": {
          "type": "object",
          "properties": {
            "tasks": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "task": {"type": "string", "description": "The name of the subtask."},
                  "description": {"type": "string", "description": "What the subtask does."}
                },
                "required": ["task", "description"]
              }
            }
          },
          "required": ["tasks"]
        }
      }
    ],
    "max_new_tokens": 256
  }
  ```
- **Response example**:

  ```json
  {
    "role": "assistant",
    "content": "{\"tasks\": [{\"task\": \"Find the book\", \"description\": \"Locate the book.\"}, {\"task\": \"Pick up the book\", \"description\": \"Lift the book.\"}, {\"task\": \"Walk to the side table\", \"description\": \"Approach the side table.\"}, {\"task\": \"Put the book on the side table\", \"description\": \"Gently place the book on the side table.\"}]}"
  }
  ```

## Memory options

### `dtype=torch.float16`
- **What it does**: computes and stores values in 16-bit floating point (FP16), halving memory compared with float32.
- **Benefit**: well suited to modern GPUs (such as NVIDIA Ampere) and speeds up inference.

### `quantization_config=BitsAndBytesConfig(load_in_8bit=True)`
- **What it does**: quantizes the model weights to 8 bits, cutting memory use substantially.
- **Benefit**: lets larger models (such as 13B or 30B) run on GPUs with limited memory.

## How it works

1. **Model loading**:
   - The base-model service loads in float32 by default; turn on 8-bit quantization if GPU memory is short (see "Out of GPU memory"). The LoRA service loads its two models in FP16 with 4-bit quantization.

2. **API endpoints**:
   - `/generate`: basic text generation
   - `/generate_with_function`: generation that uses tools

3. **Tools per request**:
   - Callers define tools with a JSON schema for each request.

4. **Response parsing**:
   - The service checks the output for a tool call and reports it in `tool_call`.

## Customize and extend

### Add tools
- Define new tools following the tool schema (`ToolSchema`).
- Extend the endpoint logic to support more tools.

### Generation parameters
- Adjust `temperature`, `top_p`, and so on to change the style of the output.

### Change the model
- Update `model_id` to use a different LLaMA model or another Hugging Face model.

## Troubleshooting

### Out of GPU memory
- **Check for other processes on the GPU**:
  ```bash
  nvidia-smi
  ```
- **Stop processes you don't need**:
  ```bash
  kill -9 <PID>
  ```
- **Lower `max_new_tokens`**, for example from 256 to 128.
- **Turn on 8-bit quantization**: uncomment this line in `llama_api_service_function.py`:
  ```python
  quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True),
  ```
- **Use a smaller model** (for example 7B instead of 13B or 30B).

## Future work
- More tools.
- Domain-specific fine-tuning.
- Multi-GPU and distributed deployment.

## License

- Code: Apache-2.0, see [LICENSE](LICENSE).
- LoRA weights and tokenizers in the `Llama-3.1-8B-Instruct-bnb-4bit_*` folders: derived from Meta Llama 3.1 and distributed under the [Llama 3.1 Community License](LICENSE-LLAMA-3.1). Use must follow the [Acceptable Use Policy](USE_POLICY-LLAMA-3.1.md); see also [NOTICE](NOTICE).

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
