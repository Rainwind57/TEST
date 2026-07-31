# 量化研究平台后端镜像（P2-11 可复现环境）
# 锁定 Python 版本 + requirements，确保「本地能跑则容器能跑」
FROM python:3.11-slim

# 时区设为 Asia/Shanghai（A 股交易日历、定时任务依赖）
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先装依赖（利用 Docker 层缓存：代码变不动依赖层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码
COPY app/ ./app/
COPY pytest.ini ./
COPY tests/ ./tests/

# 数据库与模型目录挂载点（容器销毁后数据保留）
VOLUME ["/app/quant.db", "/app/ml_models"]

EXPOSE 8000

# 健康检查（P0-4 的 /api/health 端点）
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
