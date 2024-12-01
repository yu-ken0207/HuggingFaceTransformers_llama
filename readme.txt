1. 確保主機允許外部連線

    確保防火牆允許外部訪問 FastAPI 預設的 8000 端口：

    sudo ufw allow 8000

啟動 API，綁定到所有網絡接口
    修改啟動命令
    默認情況下，FastAPI 僅綁定到 127.0.0.1（本地訪問），需要將它綁定到 0.0.0.0（所有網絡接口），讓其他電腦可以通過 IP 訪問。

    啟動命令示例：
    uvicorn script_name:app --host 0.0.0.0 --port 8000

    說明：
    --host 0.0.0.0：允許從所有網絡接口訪問 API。
    --port 8000：API 將運行在 8000 端口。



在其他電腦上訪問 API
假設主機的內部 IP 是 192.168.1.100，啟動 API 後，其他電腦可以通過以下方式訪問：
使用 curl 測試
    curl -X POST "http://192.168.1.100:8000/generate" -H "Content-Type: application/json" -d '{
        "messages": [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": "Who are you? 繁體中文回答"}
        ],
        "max_new_tokens": 256
    }'



使用瀏覽器或工具
    在瀏覽器中訪問：
    http://192.168.1.100:8000/docs
    FastAPI 提供的 Swagger UI 文檔可以測試 API。
    140.117.73.146

