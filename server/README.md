# server

D1 仅占位。计划：

- `run_gpu.sh` / `run_cpu.sh`：一行起 llama-server（D4）
- `proxy.py`：FastAPI 代理（如需统一鉴权/日志）（D4）
- `Dockerfile.cpu` / `Dockerfile.gpu` / `docker-compose.yml`（D11）

起服务的标准命令（D4 用）：

```bash
llama-server -m "$MODEL" -ngl 99 -c 4096 --host 0.0.0.0 --port 8080
curl -s http://127.0.0.1:8080/v1/models
```
