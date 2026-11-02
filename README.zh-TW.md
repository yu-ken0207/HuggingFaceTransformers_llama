# Llama API 服務（FastAPI）

[English](README.md) | **繁體中文**

[![Checks](https://github.com/yu-ken0207/HuggingFaceTransformers_llama/actions/workflows/check.yml/badge.svg)](https://github.com/yu-ken0207/HuggingFaceTransformers_llama/actions/workflows/check.yml)

**Built with Llama**：兩個 `Llama-3.1-8B-Instruct-bnb-4bit_*` 資料夾裡的 LoRA 權重由 Meta Llama 3.1 微調而來，授權見最下方。

把 Meta Llama 3.1 8B Instruct 包成 FastAPI 服務，用來做機器人任務規劃：`/generate` 是一般的對話生成，`/generate_with_function` 讓呼叫端用 JSON schema 定義工具（例如把「把書放到邊桌上」拆成步驟）。另一個服務載入兩個 LoRA 權重，分別做任務拆解和子任務規劃，權重就附在這個 repo 裡，是用 [LlamaLoRAFinetuner](https://github.com/yu-ken0207/LlamaLoRAFinetuner) 訓練的。

- 一般的文字生成（`/generate`）
- 每次請求自訂工具的生成（`/generate_with_function`）

## 功能特點

### 文字生成
- 用 `/generate` 根據訊息產生回覆。

### 每次請求自訂工具
- 用 JSON schema 定義工具，處理像拆解任務這類工作（`/generate_with_function`）。

### 大模型省記憶體
- 支援 `torch.float16` 和 8-bit 量化，減少顯示卡記憶體用量。

## 環境需求
- Python 3.11+（實測 3.12）
- 支援 CUDA 的 GPU
- 先依顯示卡的 CUDA 版本安裝 torch（其他 CUDA 版本的指令見 [PyTorch 官網](https://pytorch.org/get-started/locally/)），再裝其他套件；`environment.yml` 只建立 Python 環境：

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # CUDA 13.0
pip install -r requirements.txt
```

`requirements.txt` 是 2026 年 11 月實測過的版本（transformers 5.17.0、unsloth 2026.10.11 等），在 Linux ARM64、NVIDIA Blackwell 顯示卡、CUDA 13.0 上實測過兩個服務的 `/generate` 與 `/generate_with_function`（原版模型服務以 Llama 3.2 1B Instruct 代替 8B 測試，float32 與 8-bit 量化都跑過）。x86 電腦照同樣步驟、把 torch 換成自己 CUDA 版本的安裝指令即可，但沒有實機測過；GitHub 自動檢查會確認套件版本在 x86、Python 3.11 和 3.12 都裝得起來（版本不衝突）。

> **注意**：unsloth 目前只支援 datasets 5.0 以前的版本，跟著裝進來的 datasets 4.8.5 與 fsspec 各有一個已知漏洞，分別出在「資料夾型資料集」與「Kerchunk 參考檔」，這個服務沒有用到。這個服務也沒有驗證機制，只建議在自己的電腦或可信任的內網測試，不要直接開到網際網路上。

## 設定步驟

### 1. 環境設定
為了避免記憶體碎片、讓 GPU 用得更有效率，先設定環境變數：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 2. 啟動 API
啟動 FastAPI 服務：

原版 Llama 3.1（支援工具呼叫）。先把 `llama_api_service_function.py` 裡的 `model_id` 改成模型所在的路徑：

```bash
uvicorn llama_api_service_function:app --host 0.0.0.0 --port 8000
```

或載入這個 repo 附的兩個 LoRA 權重（任務拆解、子任務規劃，另外需要 `unsloth`）：

```bash
uvicorn llama_api_service_fintune:app --host 0.0.0.0 --port 8000
```

## API 端點說明

### 1. `/generate`
- **說明**：根據使用者的訊息產生回覆。
- **請求範例**：

  ```json
  {
    "messages": [
      {"role": "system", "content": "角色：專業的任務規劃機器人"},
      {"role": "user", "content": "將刀子和杯子移動到櫃檯。"}
    ],
    "max_new_tokens": 256
  }
  ```
- **回應範例**：

  ```json
  {
    "role": "assistant",
    "content": "1. 找到刀子。2. 拿起刀子。3. 找到杯子。4. 拿起杯子。5. 走到櫃檯。6. 將刀子和杯子放在櫃檯上。"
  }
  ```

### 2. `/generate_with_function`
- **說明**：讓呼叫端自訂工具，處理比較複雜的生成任務。
- **請求範例**：

  ```json
  {
    "messages": [
      {"role": "system", "content": "接收一個任務，將其分解成多個步驟。"},
      {"role": "user", "content": "將書本放在側桌上。"}
    ],
    "tools": [
      {
        "name": "decompose_task",
        "description": "將任務分解為子任務，並返回包含子任務列表的 JSON 對象。",
        "parameters": {
          "type": "object",
          "properties": {
            "tasks": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "task": {"type": "string", "description": "子任務的名稱。"},
                  "description": {"type": "string", "description": "子任務的描述。"}
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
- **回應範例**：

  ```json
  {
    "role": "assistant",
    "content": "{\"tasks\": [{\"task\": \"找到書本\", \"description\": \"定位書本的位置。\"}, {\"task\": \"拿起書本\", \"description\": \"將書本提起來。\"}, {\"task\": \"走到側桌\", \"description\": \"接近側桌。\"}, {\"task\": \"將書本放在側桌上\", \"description\": \"輕輕地將書本放在側桌上。\"}]}"
  }
  ```

## 省記憶體的設定

### `dtype=torch.float16`
- **作用**：用 16-bit 浮點數（FP16）計算和儲存，記憶體用量是 float32 的一半。
- **好處**：新一點的 GPU（例如 NVIDIA Ampere 架構）都支援，推論也比較快。

### `quantization_config=BitsAndBytesConfig(load_in_8bit=True)`
- **作用**：把模型權重量化成 8-bit，大幅減少記憶體用量。
- **好處**：顯示卡記憶體不夠時，也能跑比較大的模型（例如 13B、30B）。

## 程式邏輯

1. **模型初始化**：
   - 原版模型服務預設用 float32 載入，顯示卡記憶體不夠時可以打開 8-bit 量化（見「顯示卡記憶體不足」）；LoRA 服務用 FP16 加 4-bit 量化載入兩個模型。

2. **API 端點**：
   - `/generate`：基本的文字生成。
   - `/generate_with_function`：需要用到工具的任務。

3. **每次請求自訂工具**：
   - 呼叫端可以用 JSON schema 定義這次要用的工具。

4. **回應解析**：
   - 檢查生成內容裡有沒有工具呼叫，結果放在回應的 `tool_call`。

## 自訂與擴充

### 新增工具
- 照 `ToolSchema` 的格式定義新工具。
- 修改端點的程式來支援更多工具。

### 調整生成參數
- 修改 `temperature`、`top_p` 等參數，調整回答的風格。

### 更換模型
- 改 `model_id` 就能換成其他 Llama 或 Hugging Face 上的模型。

## 常見問題

### 顯示卡記憶體不足
- **看看有沒有其他程式在用 GPU**：
  ```bash
  nvidia-smi
  ```
- **關掉用不到的程式**：
  ```bash
  kill -9 <PID>
  ```
- **減少 `max_new_tokens` 的值**，例如從 256 減少到 128。
- **啟用 8-bit 量化**：把 `llama_api_service_function.py` 裡這一行的註解拿掉：
  ```python
  quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True),
  ```
- **選擇較小的模型**（如 7B 而非 13B 或 30B）。

## 未來規劃
- 支援更多工具。
- 針對特定領域微調模型。
- 支援多 GPU 和分散式部署。


## License

- 程式碼：Apache-2.0，見 [LICENSE](LICENSE)。
- `Llama-3.1-8B-Instruct-bnb-4bit_*` 資料夾裡的 LoRA 權重與 tokenizer：由 Meta Llama 3.1 衍生，依 [Llama 3.1 Community License](LICENSE-LLAMA-3.1) 散布，使用時要遵守 [Acceptable Use Policy](USE_POLICY-LLAMA-3.1.md)，另見 [NOTICE](NOTICE)。

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
