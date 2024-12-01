import requests
import json

# API 的 URL
url = "http://127.0.0.1:8000/generate"

# 定義請求的數據
payload = {
    "messages": [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you? 繁體中文回答"}
    ],
    "max_new_tokens": 256
}

# 定義 HTTP 請求的 headers
headers = {
    "Content-Type": "application/json"
}

# 發送 POST 請求
response = requests.post(url, headers=headers, data=json.dumps(payload))

# 處理響應
if response.status_code == 200:
    print("Response from API:")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
