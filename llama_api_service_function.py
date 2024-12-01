import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
import transformers
import torch

app = FastAPI()

# 啟用內存配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 清空 CUDA 緩存
torch.cuda.empty_cache()

# 模型和分詞器初始化
model_id = "/home/ailab/workspace/kxkken/Meta-Llama-3.1-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # float32
    load_in_8bit=True,  # 啟用 8-bit 量化 可以註解 將模型權重從浮點數壓縮到 8-bit  8-bit 量化只影響模型的權重存儲，而激活值仍然使用 FP16 或 FP32 進行計算
    device_map="balanced"
)
"""
torch_dtype=torch.float16 的作用
降低計算精度的內存需求：

使用 16-bit 浮點數（FP16）將模型的計算和存儲壓縮至 float32 的一半。
FP16 是現代 GPU（如 NVIDIA Ampere 架構）專門優化的一種數據格式，可以在不顯著影響推理精度的情況下提高計算速度。
針對激活值的存儲和計算：

在模型的前向和後向傳播中，激活值通常以 FP16 形式存儲並進行計算。
FP16 不會影響激活值存儲的壓縮效果，即使模型本身量化為 8-bit。
"""
"""
load_in_8bit=True 的作用
進一步壓縮模型權重：

使用 8-bit 量化技術將模型權重從浮點數壓縮為 8-bit 數據。
和 FP16 相比，8-bit 量化的內存占用進一步減少一半，從而大幅降低顯存需求。
對激活值無影響：

8-bit 量化只影響模型的權重存儲，而激活值仍然使用 FP16 或 FP32 進行計算。
"""
"""
不同部分的優化：

load_in_8bit=True 僅影響 模型權重的存儲，不影響激活值的存儲方式。
torch_dtype=torch.float16 用於 激活值的計算和存儲，與權重量化互補。
同時啟用可以使權重和激活值的內存需求都達到最優化。
平衡內存節省和精度：

單獨啟用 8-bit 量化，激活值仍以 float32 存儲，內存節省有限。
單獨啟用 FP16，權重的內存需求仍較高（FP16 比 8-bit 量化多 2 倍內存）。
同時啟用可以在 內存節省 和 計算效率 間取得最佳平衡。
計算效率最大化：

使用 FP16 計算激活值能利用 GPU 的硬件優化，顯著提高推理速度。
8-bit 量化權重在推理時仍需轉換為 FP16 或 FP32 參與計算，因此直接使用 FP16 激活值可以避免不必要的數據類型轉換。
"""

#------------------------------------------------------------------------------------------------------------
# 定義請求和回應的數據模型
class GenerateRequest(BaseModel):
    messages: list
    max_new_tokens: int = 256

class GenerateResponse(BaseModel):
    role: str
    content: str

# POST 路由：不使用函數呼叫
@app.post(
    "/generate",
    response_model=GenerateResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "messages": [
                            {"role": "system", "content": "Role: Professional Task Planning Robot\n\nGuidelines:\n- No additional text or explanations are needed.\n- Each task should be split into as many steps as needed to make it easy to follow.\n- No task can be omitted\n- Each step should be a single, clear action\n"},
                            {"role": "user", "content": "Interact with a household to solve a task. Here are two examples:\nYour task is to: Move a knife and cup to a counter.\n\n> Think: To solve the task, I need to find the knife. Then, I need to pick up the knife. Next, I need to find the cup. Then, I need to pick up the cup. Next, I need to walk to the counter. Finally, I need to place the knife and cup on the counter.\n\nPlease break down this task into a sequence of simple, executable actions.\nYour task is to: put a spray bottle on the toilet.\n\n> Think: To solve the task, I need to find the spray bottle. Then, I need to pick up the spray bottle. Next, I need to walk to the toilet. Finally, I need to place the spray bottle on the toilet.\n\nPlease break down this task into a sequence of simple, executable actions.\n\nNow, based on the examples, \nYour task is to: put some book on sidetable..\nThink: To solve the task, Your executable actions : "}
                        ],
                        "max_new_tokens": 256
                    }
                }
            }
        }
    }
)
async def generate_text(request: GenerateRequest):
    messages = request.messages
    max_new_tokens = request.max_new_tokens

    # 生成文本
    inputs = tokenizer.apply_chat_template(
        conversation=messages,  # 輸入訊息
        add_generation_prompt=True,  # 自動添加生成提示
        return_dict=True,  # 返回字典形式的輸入
        return_tensors="pt"  # 返回 PyTorch 張量
    )

    # 移動輸入到模型的設備
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # 使用模型生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # 提取生成的內容
    generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]).strip()

    return GenerateResponse(
        role="assistant",
        content=generated_text
    )

#------------------------------------------------------------------------------------------------------------

# 定義工具模式
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict

# 定義請求和回應模型
class GenerateWithFunctionRequest(BaseModel):
    messages: list
    tools: list[ToolSchema]  # 動態工具字段
    max_new_tokens: int = 256

class GenerateWithFunctionResponse(BaseModel):
    role: str
    content: str

# 定義 POST 路由
@app.post(
    "/generate_with_function",
    response_model=GenerateWithFunctionResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You will receive a task. Please decompose the task into multiple steps and return them in a structured format, "
                                    "ensuring the output follows the JSON structure below:\n"
                                    "{\n"
                                    "  \"tasks\": [\n"
                                    "    {\n"
                                    "      \"task\": \"<task>\",\n"
                                    "      \"description\": \"<description>\"\n"
                                    "    },\n"
                                    "    ...\n"
                                    "  ]\n"
                                    "}\n"
                                    "Do not include any additional text or explanations."
                                )
                            },
                            {"role": "user", "content": "To solve the task \"put some book on sidetable\", here's the sequence of simple, executable actions:\n\n1. Find the book.\n2. Pick up the book.\n3. Walk to the sidetable.\n4. Place the book on the sidetable.<|eot_id|>"}
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
                                                    "task": {
                                                        "type": "string",
                                                        "description": "The name of the subtask."
                                                    },
                                                    "description": {
                                                        "type": "string",
                                                        "description": "Description of the subtask."
                                                    }
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
                }
            }
        }
    }
)
async def decompose_task(request: GenerateWithFunctionRequest):
    messages = request.messages
    tools = request.tools
    max_new_tokens = request.max_new_tokens

    # 將工具轉換為符合 apply_chat_template 要求的格式
    tools_json = [tool.dict() for tool in tools]

    # 構建輸入
    inputs = tokenizer.apply_chat_template(
        conversation=messages,
        tools=tools_json,  # 動態工具傳遞
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # 使用模型生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # 提取生成文本
    generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]).strip()

    # 嘗試解析工具調用
    tool_used = False
    tool_call = None
    try:
        if "<tool_call>" in generated_text and "</tool_call>" in generated_text:
            tool_call_start = generated_text.find("<tool_call>") + len("<tool_call>")
            tool_call_end = generated_text.find("</tool_call>")
            tool_call_json = generated_text[tool_call_start:tool_call_end].strip()
            tool_call = json.loads(tool_call_json)

            if "name" in tool_call:
                tool_used = True
    except Exception as e:
        # 捕獲解析錯誤
        tool_call = None

    # 如果檢測到工具調用，執行工具函數
    if tool_used:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})
        if tool_name == "decompose_task":
            result = [{"task": t["task"], "description": t["description"]} for t in tool_args.get("tasks", [])]
            return GenerateWithFunctionResponse(
                role="assistant",
                content=json.dumps({"tasks": result})
            )

    # 無工具調用，返回原始生成結果
    return GenerateWithFunctionResponse(
        role="assistant",
        content=generated_text
    )
#------------------------------------------------------------------------------------------------------------
