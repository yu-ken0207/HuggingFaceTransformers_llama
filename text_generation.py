import transformers
import torch

model_id = "/home/ailab/workspace/kxkken/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},#float16 float32
    device_map="auto", # auto balanced
    # max_memory={0: "10GiB"},  # 為每張 GPU 分配內存
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you? 繁體中文回答"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])