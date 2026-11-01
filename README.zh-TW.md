# 基於 FastAPI 和 LLaMA 模型的 API 服務

[English](README.md) | **繁體中文**

[![Checks](https://github.com/yu-ken0207/HuggingFaceTransformers_llama/actions/workflows/check.yml/badge.svg)](https://github.com/yu-ken0207/HuggingFaceTransformers_llama/actions/workflows/check.yml)

**Built with Llama**：兩個 `Llama-3.1-8B-Instruct-bnb-4bit_*` 資料夾裡的 LoRA 權重由 Meta Llama 3.1 微調而來，授權見最下方。

此專案展示如何構建基於 FastAPI 的 API 服務，並整合 Meta LLaMA 3.1 模型進行文本生成任務。服務包括：

- 簡單的文本生成端點（`/generate`）。
- 支援工具的生成端點（`/generate_with_function`），動態處理複雜任務的工具。

## 功能特點

### 文本生成
- 使用 `/generate` 端點生成回應。

### 動態工具整合
- 使用 JSON schema 定義工具，完成如分解任務的複雜功能（`/generate_with_function`）。

### 優化大型模型的內存管理
- 支援 `torch.float16` 精度和 8-bit 量化以提高運行效率。

## 環境需求
- Python 3.11+（實測 3.12）
- 支援 CUDA 的 GPU
- 先依顯示卡的 CUDA 版本安裝 torch（其他 CUDA 版本的指令見 [PyTorch 官網](https://pytorch.org/get-started/locally/)），再裝其他套件；`environment.yml` 只建立 Python 環境：

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # CUDA 13.0
pip install -r requirements.txt
```

`requirements.txt` 是 2026 年 11 月實測過的版本（transformers 5.17.0、unsloth 2026.10.11 等），在 Linux ARM64、NVIDIA Blackwell 顯示卡、CUDA 13.0 上實測過兩個服務的 `/generate` 與 `/generate_with_function`（原版模型服務以 Llama 3.2 1B Instruct 代替 8B 測試，float32 與 8-bit 量化都跑過）。x86 電腦照同樣步驟、把 torch 換成自己 CUDA 版本的安裝指令即可，但沒有實機測過；GitHub 自動檢查會確認套件版本在 x86、Python 3.11 和 3.12 都解得開。

> **注意**：unsloth 目前只支援 datasets 5.0 以前的版本，跟著裝進來的 datasets 4.8.5 與 fsspec 各有一個已知漏洞，分別出在「資料夾型資料集」與「Kerchunk 參考檔」，這個服務沒有用到。這個服務也沒有驗證機制，只建議在自己的電腦或可信任的內網測試，不要直接開到網際網路上。

## 設置步驟

### 1. 環境配置
為避免內存碎片化並優化 GPU 使用，設置環境變量：

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
- **說明**：根據用戶提供的訊息生成回應。
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
- **說明**：允許用戶定義工具以完成更複雜的生成任務。
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

## 內存優化機制

### `dtype=torch.float16`
- **作用**：使用 16-bit 浮點數（FP16）進行計算和存儲，降低內存需求。
- **優勢**：適用於現代 GPU（如 NVIDIA Ampere 架構），能加速推理過程。

### `quantization_config=BitsAndBytesConfig(load_in_8bit=True)`
- **作用**：將模型權重量化為 8-bit 數據，顯著降低內存占用。
- **優勢**：可在顯存受限的 GPU 上運行大模型（如 13B、30B）。

### 結合使用的原因
- **權重和激活值的最佳優化**：`torch.float16` 用於激活值存儲和計算，8-bit 量化用於模型權重存儲。
- **平衡內存和性能**：同時啟用能在內存節省與計算效率間取得最佳平衡。

## 程式邏輯

1. **模型初始化**：
   - 原版模型服務預設用 float32 載入，顯存不夠時可以打開 8-bit 量化（見「顯存不足」）；LoRA 服務用 FP16 加 4-bit 量化載入兩個模型。

2. **API 端點**：
   - `/generate`：處理基本文本生成任務。
   - `/generate_with_function`：處理需要工具支持的任務。

3. **動態工具支持**：
   - 用戶可通過 JSON schema 定義工具，用於特定的生成需求。

4. **回應解析**：
   - 檢查生成內容裡有沒有工具呼叫，結果放在回應的 `tool_call`。

## 自訂與擴展

### 新增工具
- 按照工具模式（ToolSchema）定義新工具。
- 更新端點邏輯以支持更多的工具功能。

### 調整生成參數
- 修改 `temperature`、`top_p` 等參數以控制生成風格。

### 更換模型
- 更新 `model_id` 以支持不同的 LLaMA 模型或其他 Hugging Face 模型。

## 常見問題與解決方案

### 顯存不足
- **檢查是否有其他進程佔用 GPU**：
  ```bash
  nvidia-smi
  ```
- **終止不必要的進程**：
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
- 增加更多工具支持。
- 實現領域專用的模型微調。
- 優化多 GPU 或分佈式部署場景。

這套服務結合了 FastAPI 和 LLaMA 模型的優勢，是構建高效推理 API 的理想解決方案，具有靈活性和擴展性。


## License

- 程式碼：Apache-2.0，見 [LICENSE](LICENSE)。
- `Llama-3.1-8B-Instruct-bnb-4bit_*` 資料夾裡的 LoRA 權重與 tokenizer：由 Meta Llama 3.1 衍生，依 [Llama 3.1 Community License](LICENSE-LLAMA-3.1) 散布，使用時要遵守 [Acceptable Use Policy](USE_POLICY-LLAMA-3.1.md)，另見 [NOTICE](NOTICE)。

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
