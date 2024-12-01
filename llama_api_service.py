from fastapi import FastAPI
from pydantic import BaseModel
import transformers
import torch

app = FastAPI()

model_id = "/home/ailab/workspace/kxkken/Meta-Llama-3.1-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation", # 指定管線的任務類型，例如 "text-generation"、"text-classification" 等 如 "text-generation"、"fill-mask"、"question-answering" 
    model=model_id,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="balanced",
)

class GenerateRequest(BaseModel):
    messages: list
    max_new_tokens: int = 256

class GenerateResponse(BaseModel):
    role: str
    content: str

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

    # 使用管線生成文本
    outputs = pipeline(
        messages,
        max_new_tokens=max_new_tokens, # 生成的最大新標記數（不包括輸入的長度）。整數值。
        return_full_text=False,  # 只返回生成的文本部分 不會回應完整內容
    )

    # 提取生成的字符串內容
    generated_text = outputs[0]["generated_text"].strip()  # 提取生成的純字符串
     # 返回特定格式的回應
    return GenerateResponse(
        role="assistant",
        content=generated_text
    )



# 啟動方式：
# 使用 `uvicorn` 啟動 API，如：
# uvicorn script_name:app --reload

# uvicorn llama_api_service:app --reload
# uvicorn llama_api_service:app --host 0.0.0.0 --port 8000
# http://140.117.73.146:8000/docs



        # max_length 生成文本的最大總長度（包括輸入和生成的文本）。整數值。
        # min_length 生成文本的最小長度。整數值。
        # do_sample 是否使用隨機取樣來生成文本。True 或 False，預設為 True。
        # num_beams 使用的束搜索數量，適用於提高生成質量。整數值，1 代表貪婪搜索。
        # temperature 控制取樣的隨機性，值越高，生成結果越隨機 浮點數，通常在 0.7 到 1.0 之間。
        # repetition_penalty 重複懲罰，避免生成重複的內容。大於 0 的浮點數。
        # length_penalty 長度懲罰，控制生成文本的長度偏好。可選值：浮點數。
        # early_stopping 在所有束搜索完成前停止生成。可選值：True 或 False。
        # num_return_sequences 生成的文本數量。可選值：整數值，預設為 1。
        # return_full_text 是否返回完整的文本（包括輸入的部分）。可選值：True 或 False。
        # bad_words_ids 指定禁止出現在生成結果中的詞語。可選值：詞語 ID 的列表。
        # pad_token_id、eos_token_id：描述：指定填充標記和結束標記的 ID。可選值：整數值。