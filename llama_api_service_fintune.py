import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
import transformers
import torch
from unsloth import FastLanguageModel

app = FastAPI()

# 啟用內存配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 清空 CUDA 緩存
torch.cuda.empty_cache()

#------------------------------------------------------------------------------------------------------------
# SubTaskPlanner模型
model_TaskDecomposer = "Llama-3.1-8B-Instruct-bnb-4bit_TaskDecomposer"
model_TaskDecomposer, tokenizer_TaskDecomposer = FastLanguageModel.from_pretrained(
        model_name=model_TaskDecomposer,
        max_seq_length=512,  # 設定最大序列長度
        dtype="float16",  # 使用 FP16
        load_in_4bit=True,  # 加載 4-bit 量化模型
    )
# 啟用推理模式
FastLanguageModel.for_inference(model_TaskDecomposer)

# SubTaskPlanner模型
model_id = "Llama-3.1-8B-Instruct-bnb-4bit_SubTaskPlanner"
model_SubTaskPlanner, tokenizer_SubTaskPlanner = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=512,  # 設定最大序列長度
        dtype="float16",  # 使用 FP16
        load_in_4bit=True,  # 加載 4-bit 量化模型
    )
# 啟用推理模式
FastLanguageModel.for_inference(model_SubTaskPlanner)
#------------------------------------------------------------------------------------------------------------
# 定義請求和回應的數據模型
class GenerateRequest(BaseModel):
    messages: list
    max_new_tokens: int = 512

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
                            {"role": "system", "content": "You are a helpful AI assistant."},
                            {"role": "user", "content": "Instruction: 以下是一個總體任務描述和其對應的高階步驟。請學習總體任務與高階步驟的關係 並根據高階步驟生成總體任務描述\n\n Input: put some book on sidetable.\n\n Response: "}
                        ],
                        "max_new_tokens": 512
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
    inputs = tokenizer_TaskDecomposer.apply_chat_template(
        conversation=messages,  # 輸入訊息
        add_generation_prompt=True,  # 自動添加生成提示
        return_dict=True,  # 返回字典形式的輸入
        return_tensors="pt"  # 返回 PyTorch 張量
    )

    # 移動輸入到模型的設備
    inputs = {k: v.to(model_TaskDecomposer.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # 使用模型生成
    outputs = model_TaskDecomposer.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # 提取生成的內容
    generated_text = tokenizer_TaskDecomposer.decode(outputs[0][len(inputs["input_ids"][0]):]).strip()

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
    max_new_tokens: int = 512

class GenerateWithFunctionResponse(BaseModel):
    role: str
    content: str
    tool_call:bool

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
                                                },
                                                "required": ["task"]
                                            }
                                        }
                                    },
                                    "required": ["tasks"]
                                }
                            }
                        ],
                        "max_new_tokens": 512
                    }
                }
            }
        }
    }
)
async def generate_text_func(request: GenerateWithFunctionRequest):
    messages = request.messages
    tools = request.tools
    max_new_tokens = request.max_new_tokens

    # 將工具轉換為符合 apply_chat_template 要求的格式
    tools_json = [tool.dict() for tool in tools]

    # 構建輸入
    inputs = tokenizer_SubTaskPlanner.apply_chat_template(
        conversation=messages,
        tools=tools_json,  # 動態工具傳遞
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model_SubTaskPlanner.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # 使用模型生成
    outputs = model_SubTaskPlanner.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # 提取生成文本
    generated_text = tokenizer_SubTaskPlanner.decode(outputs[0][len(inputs["input_ids"][0]):])

    if "<|python_tag|>" in generated_text:
        return GenerateWithFunctionResponse(
            role="assistant",
            content=generated_text,
            tool_call= True
        )

    # 無工具調用，返回原始生成結果
    return GenerateWithFunctionResponse(
        role="assistant",
        content=generated_text,
        tool_call= False
    )